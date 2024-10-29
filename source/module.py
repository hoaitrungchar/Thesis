from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch import SwinUnet
from torch.nn.modules.loss import CrossEntropyLoss
# from pytorch_lightning.metrics import functional as FM

class MaskedPredictModule(LightningModule):
    def __init__(self, 
                 net: SwinUnet,
                pretrained_path: str = "",
                lr: float = 5e-4,
                beta_1: float = 0.9,
                beta_2: float = 0.99,
                weight_decay: float = 1e-5
                ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.hparams.lr=lr
        self.hparams.beta_1=beta_1
        self.hparams.beta_2=beta_2
        self.hparams.weight_decay=weight_decay
        self.it = 0
        self.ce_loss = CrossEntropyLoss()
        #self.dice_loss = DiceLoss(n_classes=1)
        self.bce_loss = F.binary_cross_entropy_with_logits
    def training_step(self,batch):
        input ,masked,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        # print(input.shape)
        # print(masked.shape)
        # print(groudtruth.shape)
        predicted= self.net(input)

        loss = self.bce_loss(predicted, masked) #+ 0.6*self.dice_loss(predicted, masked)
        
        self.log("train/loss", loss)
        return loss.float()


    def test_step(self,batch, batch_idx):
        input ,masked,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        predicted= self.net(input)
        loss = self.bce_loss(predicted, masked) #+ 0.6*self.dice_loss(predicted, masked)
        #self.log("test/mse", mse_loss)
        self.log("test/loss", loss)
        return loss.float()

        
    def validation_step(self,batch, batch_idx):
        input ,masked,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        predicted= self.net(input)
        loss = self.bce_loss(predicted, masked) #+ 0.6*self.dice_loss(predicted, masked)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(), 
                               lr=self.hparams.lr,
                               betas=[self.hparams.beta_1,self.hparams.beta_2],
                               weight_decay=self.hparams.weight_decay
                                      )