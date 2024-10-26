import yaml
import torch
import pytorch_lightning as pl
import arch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_lightning import LightningModule

class MaskedPredictModule(LightningModule):
    def __init__(self, 
                 net: arch.SwinUnet
                ):
        super().__init__()
        self.net = net


# Load configuration from YAML file
def load_config(yaml_path):
    with open(yaml_path, 'r',  encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def load_checkpoint_from_yaml(config_path, checkpoint_path):
    # Load YAML config
    config = load_config(config_path)
    dict=config.get('model')['net']
    
    checkpoint = torch.load(checkpoint_path)
    net=arch.SwinUnet(
        config='',
        patch_size=dict['patch_size'],
        num_classes=dict['num_classes'],
        embed_dim=dict['embed_dim'],
        depths=dict['depths'],
        depths_decoder=dict['depths_decoder'],
        num_heads=dict['num_heads'],
        window_size=dict['window_size'],
        qkv_bias=dict['qkv_bias'],
        in_chans=dict['in_chans'],
        qk_scale=dict['qk_scale'],
        drop_rate=dict['drop_rate'],
        drop_path_rate=dict['drop_path_rate'],
        ape=dict['ape'],
        patch_norm=dict['patch_norm'],
        use_checkpoint=dict['use_checkpoint']
    )

    model=MaskedPredictModule(net)
    model.load_state_dict(checkpoint['state_dict'])
    return model



model=load_checkpoint_from_yaml(
    '/home/vndata/trung/Swin_UnetFFHQ/source/config/SwinUnet.yaml',
    '/home/vndata/trung/Swin_UnetFFHQ/source/checkpoint/SwinUnet/checkpoints/epoch_029.ckpt'
    )
img=cv2.imread('/home/vndata/trung/ffhq-dataset/images1024x1024/04000/04000.png')
img_not_resize=img
img =cv2.resize(img,(256,256))


mask=cv2.imread('/home/vndata/testing_mask_dataset/00004.png',cv2.IMREAD_GRAYSCALE)
mask=cv2.bitwise_not(mask)
mask = cv2.resize(mask, (256, 256))
mask=mask/255

mask= np.where(mask<0.5, 0, 1)
print(mask)
mean,std=np.load('/home/vndata/trung/ffhq-dataset/images1024x1024/mean.npz'),np.load('/home/vndata/trung/ffhq-dataset/images1024x1024/std.npz')
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean['arr_0'],std['arr_0']
                             )
    ])
# img=transform(img)
transform_img = transforms.ToPILImage()
to_tensor=transforms.Compose([
        transforms.ToTensor()])
img=to_tensor(img)
img_bf_mask=transform_img(img)
img_not_resize=transform_img(img_not_resize)
img_not_resize.save('/home/vndata/trung/Swin_UnetFFHQ/source/resize.png')
img_bf_mask.save('/home/vndata/trung/Swin_UnetFFHQ/source/input_bf_mask.jpg')
mask= to_tensor(mask)
img_masked=img*mask
img_masked=transform_img(img_masked)
img_masked.save('/home/vndata/trung/Swin_UnetFFHQ/source/input.jpg')
img_masked=img_masked.unsqueeze(0)
with torch.no_grad():
    predict=model.net(img_masked)

predict=predict*255
predict=predict.squeeze(0)
result_img=transform_img(predict)

print('predicted_image')
result_img.save('/home/vndata/trung/Swin_UnetFFHQ/source/abcdefgh.jpg')
