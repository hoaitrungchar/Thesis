from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from pytorch_lightning import LightningDataModule, LightningModule
import pandas as pd
import random
import os
from torchvision import transforms  
import torchvision
from typing import Tuple, Dict, List
import cv2
from generate_mask import create_mask

class DatasetReader(Dataset):
    def __init__(self,
                name_dataset: str,
                type_dataset: str,
                mask_path_test: str,
                path: str):
        
        self.name_dataset=name_dataset
        self.type_dataset=type_dataset
        self.path=path
        self.mask_path_test=mask_path_test
        self.mean=np.load(os.path.join(self.path,"mean.npz"))
        self.std=np.load(os.path.join(self.path,"std.npz"))
        if self.name_dataset =="Places2":
            path=os.path.join(path,self.type_dataset)
            if self.type_dataset in ['test', 'val']:
                self.list_image=list(map(lambda x: os.path.join(path,x),os.listdir(path)))
            else:
                self.list_image=[]
                for x in os.listdir(path):
                    x=os.path.join(path,x)
                    # print(x, flush=True)
                    for y in os.listdir(x):
                        y=os.path.join(x,y)
                        for z in os.listdir(y):
                            z=os.path.join(y,z)
                            if os.path.isfile(z):
                                self.list_image.append(z)
                            else:
                                for image in os.listdir(z):
                                    image=os.path.join(z,image)
                                    self.list_image.append(image)
        elif self.name_dataset=="FFHQ":
            list_folder_path=list(map(lambda x: os.path.join(path,x),os.listdir(path)))
            list_image_path=[]
            print(list_folder_path)
            for folder in list_folder_path:
                print(folder)
                if not os.path.isdir(folder):
                    continue
                list_image_path+=list(map(lambda x: os.path.join(folder,x),os.listdir(folder)))
            num_img=len(list_image_path)
            print(num_img)
            if type_dataset=="train":
                self.list_image=list_image_path[0:int(0.7*num_img -1)]
            elif type_dataset=="test":
                self.list_image=list_image_path[int(0.7*num_img):-1]
            elif type_dataset =="val":
                self.list_image=list_image_path[int(0.7*num_img):int(0.8*num_img-1)]
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean['arr_0'],self.std['arr_0']),
            torchvision.transforms.Resize((256,256))
        ])
        if self.type_dataset == "test":
            list_img=os.listdir(self.mask_path_test)
            self.list_mask_image=list(map(lambda x: os.path.join(self.mask_path_test,x),list_img))
            self.num_mask_image=len(self.list_mask_image)
        # print(self.type_dataset,self.list_image[0])




    def __len__(self):
        return len(self.list_image)
    
    def __getitem__(self, index:int) -> Tuple[torch.tensor, torch.tensor]:
        img=torchvision.io.read_image(self.list_image[index])
        if self.type_dataset == "test":
            n=random.randint(0, self.num_mask_image-1)
            mask=cv2.imread(self.list_mask_image[n],cv2.IMREAD_GRAYSCALE)
        else:
            mask=create_mask()
        mask=cv2.bitwise_not(mask)
        mask = cv2.resize(mask, (256, 256))
        mask=mask/255
        mask= np.where(mask<0.5, 0, 1)


        # img_masked=cv2.bitwise_and(img,img,mask=mask)
        img=self.transform(img)
        # img_masked=self.transform(img_masked)
        to_tensor=transforms.Compose([
            transforms.ToTensor()])
        mask= to_tensor(mask)
        img_masked=img*mask
        return img_masked, mask, img

    
class DataModule(LightningDataModule):
    def __init__(self,
                 name_dataset:str,
                 path:str,
                 mask_path_test:str,
                batch_size:int ,
                num_workers:int ,
                pin_memory: bool ,
                 ):
        super().__init__()
        self.hparams.path = path
        self.hparams.name_dataset=name_dataset
        self.hparams.mask_path_test=mask_path_test
        self.hparams.batch_size=batch_size
        self.hparams.num_workers=num_workers
        self.hparams.pin_memory=pin_memory

    def prepare_data(self):
        pass
    
    def setup(self,stage):
        self.train_data = DatasetReader(
            name_dataset=self.hparams.name_dataset,
            type_dataset="train",
            mask_path_test=self.hparams.mask_path_test,
            path=self.hparams.path
        )

        self.test_data= DatasetReader(
            name_dataset=self.hparams.name_dataset,
            type_dataset="test",
            mask_path_test=self.hparams.mask_path_test,
            path=self.hparams.path

        )
        
        self.val_data= DatasetReader(
            name_dataset=self.hparams.name_dataset,
            type_dataset="val",
            mask_path_test=self.hparams.mask_path_test,
            path=self.hparams.path
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

   
    