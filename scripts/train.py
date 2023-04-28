import numpy as np
import torch
import os
import argparse


from model import Unet
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import torch.nn as nn
from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def main(args):

    image_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
    ])
    dataset = OxfordIIITPet(root='/home/patel.purvi/data/',download=True, target_types='segmentation',
                                     transform=image_transform, target_transform=target_transform) 
    train_len = int(len(dataset)*0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)


    model = Unet(num_classes=3, bilinear=True, dropout=0.4)

    model.apply(weights_init_kaiming)
    wand_logger = WandbLogger(project='Equivariant-Unet')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, 'checkpoints'),
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=True,
    )

    
    trainer = Trainer(accelerator='auto', max_epochs=40, logger=wand_logger, callbacks=[checkpoint_callback, stop_callback])
    trainer.fit(model, train_loader, val_loader)




if __name__ ==  '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--log_dir', type=str, default='/scratch/patel.purvi/Unet/logs')
    
    args = arg.parse_args()
    print(args.log_dir)
    
    main(args)