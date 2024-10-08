from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch import SwinUnet
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
        

    def training_step(self,batch):
        input ,masked,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        # print(input.shape)
        # print(masked.shape)
        # print(groudtruth.shape)
        predicted= self.net(input)
        mse_loss=F.mse_loss(masked,predicted)
        self.log("training loss", mse_loss)
        return mse_loss.float()


    def test_step(self,batch):
        input ,masked,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        predicted= self.net(input)
        mse_loss=F.mse_loss(masked,predicted)
        self.log("test loss", mse_loss)
        return mse_loss.float()

        
    def validation_step(self,batch):
        input ,masked,groudtruth = batch
        input=input.float()
        masked=masked.float()
        groundtruth=groudtruth.float()
        predicted= self.net(input)
        mse_loss=F.mse_loss(masked,predicted)
        self.log("validating loss", mse_loss)
        return mse_loss

    def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(), 
                               lr=self.hparams.lr,
                               betas=[self.hparams.beta_1,self.hparams.beta_2],
                               weight_decay=self.hparams.weight_decay
                                      )