from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.SwinUnet import SwinUnet
from torch.nn.modules.loss import CrossEntropyLoss
from net.gaussian_diffusion import GaussianDiffusion
from net.UniformSampler import UniformSampler
from net.SimpleUnet import SimpleUnet
from PIL import Image
import numpy as np
from net.Unet import UNetModel
import net.gaussian_diffusion as gd
from net.respace import SpacedDiffusion,space_timesteps

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
                num_timesteps: int =2000,
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
            steps=num_timesteps,
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
        self.it=0
        
    def training_step(self,batch):
        return 0


    def test_step(self,batch, batch_idx):
        print(self.num_timesteps)
        inputs,maskeds,priors,groudtruths = batch
        std = torch.Tensor([0.5,0.5,0.5]).view(3,1,1).to(self.device)
        mean = torch.Tensor([0.5,0.5,0.5]).view(3,1,1).to(self.device)
        init_mask = self.net_init_mask(inputs)
        init_prior = self.net_init_prior(inputs)
        print('maskeds',maskeds)
        print('priors ',priors)
        print('input',inputs)
        print(groudtruths)
        input = self.training_method.q_sample(
            x_start = inputs,
            t = torch.tensor([self.num_timesteps-1,]*inputs.shape[0])
        )

        terms = self.training_method.p_sample_loop_progressive(
            model = self.net,
            shape = input.shape,
            y0=input,
            noise = None,
            clip_denoised = True,
            model_kwargs = {'mask': init_mask, 'prior': init_prior},
            device = f"cuda",
            progress=False
        )
        it = self.it
        for input in inputs:
            input = input * std + mean
            input = input *255
            input = input.detach().cpu().numpy().astype(np.uint8)
            input = input.transpose(1, 2, 0)
            input = Image.fromarray((input))
            input.save(f'/home/vndata/trung/output/input_{self.it}.png')
            it+=1

        it = self.it
        for mask in maskeds:
            mask *= 255
            mask = mask.repeat(3, 1, 1)
            mask = mask.permute(1, 2, 0).detach().cpu().numpy()
            mask = Image.fromarray((mask).astype(np.uint8))
            mask.save(f'/home/vndata/trung/output/mask_{self.it}.png')
            it+=1


        it = self.it
        for prior in priors:
            prior *= 255
            prior = prior.repeat(3, 1, 1) 
            prior = prior.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            prior = Image.fromarray((prior))
            prior.save(f'/home/vndata/trung/output/prior_{self.it}.png')
            it+=1

        it = self.it
        for groundtruth in groudtruths:
            groundtruth = groundtruth * std + mean
            groundtruth=groundtruth*255
            groundtruth = groundtruth.detach().cpu().numpy().astype(np.uint8)
            groundtruth = groundtruth.transpose(1, 2, 0)
            groundtruth = Image.fromarray((groundtruth))
            groundtruth.save(f'/home/vndata/trung/output/groundtruth_{self.it}.png')
            it+=1

        it = self.it
        for term in terms:
            print(term)
            img_predicted = term['sample']
            img_predicted = img_predicted *255
            img_predicted = torch.squeeze(img_predicted)
            img_predicted = img_predicted[0:3,:,:]
            img_predicted = img_predicted * std + mean
            img_predicted = img_predicted.detach().cpu().numpy().astype(np.uint8)
            img_predicted = img_predicted.transpose(1, 2, 0)
            img_predicted = Image.fromarray((img_predicted))
            img_predicted.save(f'/home/vndata/trung/output/image_predicted_{it}.png')

            mask_predicted = term['mask']
            mask_predicted = torch.squeeze(mask_predicted)
            mask_predicted = torch.sigmoid(mask_predicted)
            mask_predicted = mask_predicted * 255
            mask_predicted = mask_predicted.repeat(3, 1, 1)
            mask_predicted = mask_predicted.permute(1, 2, 0).detach().cpu().numpy()
            mask_predicted = Image.fromarray((mask_predicted).astype(np.uint8))
            mask_predicted.save(f'/home/vndata/trung/output/mask_predicted_{it}.png')

            prior_predicted = term['prior']
            prior_predicted = torch.squeeze(prior_predicted)
            prior_predicted = torch.sigmoid(prior_predicted)
            prior_predicted = prior_predicted * 255
            prior_predicted = prior_predicted.repeat(3, 1, 1)
            prior_predicted = prior_predicted.permute(1, 2, 0).detach().cpu().numpy()
            prior_predicted = Image.fromarray((prior_predicted).astype(np.uint8))
            prior_predicted.save(f'/home/vndata/trung/output/prior_predicted_{it}.png')
            it+=1
        self.it = it
        return 0
        
    def validation_step(self,batch, batch_idx):
        return 0
    
    
    def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(), 
                               lr=self.hparams.lr,
                               betas=[self.hparams.beta_1,self.hparams.beta_2],
                               weight_decay=self.hparams.weight_decay
                                      )
    
