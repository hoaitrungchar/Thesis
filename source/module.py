from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.SwinUnet import SwinUnet
from torch.nn.modules.loss import CrossEntropyLoss
from net.gaussian_diffusion import GaussianDiffusion
from net.UniformSampler import UniformSampler
from net.SimpleUnet import SimpleUnet
from net.Unet import UNetModel
import net.gaussian_diffusion as gd
from net.respace import SpacedDiffusion,space_timesteps
# class MaskedPredictModule(LightningModule):
#     def __init__(self, 
#                  net: SwinUnet,
#                 pretrained_path: str = "",
#                 lr: float = 5e-4,
#                 beta_1: float = 0.9,
#                 beta_2: float = 0.99,
#                 weight_decay: float = 1e-5,
#                 **kwargs
#                 ):
#         super().__init__()
#         self.save_hyperparameters(logger=False, ignore=["net"])
#         self.net = net
#         self.hparams.lr=lr
#         self.hparams.beta_1=beta_1
#         self.hparams.beta_2=beta_2
#         self.hparams.weight_decay=weight_decay
#         self.it = 0
#         self.ce_loss = CrossEntropyLoss()
#         #self.dice_loss = DiceLoss(n_classes=1)
#         self.bce_loss = F.binary_cross_entropy_with_logits
#     def training_step(self,batch):
#         input ,masked,groudtruth = batch
#         input=input.float()
#         masked=masked.float()
#         groundtruth=groudtruth.float()
#         # print(input.shape)
#         # print(masked.shape)
#         # print(groudtruth.shape)
#         predicted= self.net(input)

#         loss = self.bce_loss(predicted, masked) #+ 0.6*self.dice_loss(predicted, masked)
        
#         self.log("train/loss", loss)
#         return loss.float()


#     def test_step(self,batch, batch_idx):
#         input ,masked,groudtruth = batch
#         input=input.float()
#         masked=masked.float()
#         groundtruth=groudtruth.float()
#         predicted= self.net(input)
#         loss = self.bce_loss(predicted, masked) #+ 0.6*self.dice_loss(predicted, masked)
#         #self.log("test/mse", mse_loss)
#         self.log("test/loss", loss)
#         return loss.float()

        
#     def validation_step(self,batch, batch_idx):
#         input ,masked,groudtruth = batch
#         input=input.float()
#         masked=masked.float()
#         groundtruth=groudtruth.float()
#         predicted= self.net(input)
#         loss = self.bce_loss(predicted, masked) #+ 0.6*self.dice_loss(predicted, masked)
#         self.log("val/loss", loss)
#         return loss

#     def configure_optimizers(self):
#        return torch.optim.Adam(self.parameters(), 
#                                lr=self.hparams.lr,
#                                betas=[self.hparams.beta_1,self.hparams.beta_2],
#                                weight_decay=self.hparams.weight_decay
#                                       )

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

class DenoisedModule(LightningModule):
    def __init__(self, 
                 net: UNetModel ,
                net_init_mask : SimpleUnet,
                net_init_prior: SimpleUnet,
                pretrained_path: str = "",
                num_timesteps: int =500,
                lr: float = 5e-4,
                beta_1: float = 0.9,
                beta_2: float = 0.99,
                weight_decay: float = 1e-5,
                learn_sigma=False,
                sigma_small=False,
                noise_schedule="linear",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
                timestep_respacing="",
                **kwargs
                ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.hparams.lr=lr
        self.hparams.beta_1=beta_1
        self.hparams.beta_2=beta_2
        self.hparams.weight_decay=weight_decay
        self.net_init_mask = net_init_mask
        self.net_init_prior = net_init_prior
        self.num_timesteps = num_timesteps
        self.net = net
        self.sampler= UniformSampler(self.num_timesteps)
        self.pretrained_path = pretrained_path
        self.training_method= create_gaussian_diffusion(
            steps=self.num_timesteps,
            learn_sigma=learn_sigma,
            sigma_small=sigma_small,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=timestep_respacing,
            )
        
        self.bce_loss = F.binary_cross_entropy_with_logits
        
    def training_step(self,batch):
        input,masked,prior,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        init_mask=self.net_init_mask(input)
        init_prior=self.net_init_prior(input)
        input = torch.cat((input,init_mask,init_prior), dim =1)
        B,N,H,W=input.shape
        indicies, weight=self.sampler.sample(B)
        term = self.training_method.training_losses(self.net, input ,indicies)
        loss = term['loss'].mean()
        prior_loss= self.bce_loss(term['mask'],masked)
        mask_loss= self.bce_loss(term['prior'],prior)
        self.log("train/loss", loss+prior_loss+mask_loss)
        return loss


    def test_step(self,batch, batch_idx):
        input,masked,prior,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        init_mask=self.net_init_mask(input)
        init_prior=self.net_init_prior(input)
        input = torch.cat((input,init_mask,init_prior), dim =1)
        B,N,H,W=input.shape
        indicies, weight=self.sampler.sample(B)
        term = self.training_method.training_losses(self.net, input ,indicies)
        loss = term['loss'].mean()
        prior_loss= self.bce_loss(term['mask'],masked)
        mask_loss= self.bce_loss(term['prior'],prior)
        self.log("test/loss", loss+prior_loss+mask_loss)
        return loss
        
    def validation_step(self,batch, batch_idx):
        input,masked,prior,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        init_mask=self.net_init_mask(input)
        init_prior=self.net_init_prior(input)
        input = torch.cat((input,init_mask,init_prior), dim =1)
        B,N,H,W=input.shape
        indicies, weight=self.sampler.sample(B)
        term = self.training_method.training_losses(self.net, input ,indicies)
        loss = term['loss'].mean()
        prior_loss= self.bce_loss(term['mask'],masked)
        mask_loss= self.bce_loss(term['prior'],prior)
        self.log("val/loss", loss+prior_loss+mask_loss)
        return loss 
    
    def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(), 
                               lr=self.hparams.lr,
                               betas=[self.hparams.beta_1,self.hparams.beta_2],
                               weight_decay=self.hparams.weight_decay
                                      )
    