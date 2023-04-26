import numpy as np
import torch
import os

from model import Unet
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import argparse


def main(args):

    image_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
    ])
    dataset = OxfordIIITPet(root='../../data/',download=True, target_types='segmentation',
                                     transform=image_transform, target_transform=target_transform) 
    train_len = int(len(dataset)*0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)


    model = Unet(num_classes=37, bilinear=True, dropout=0)

    
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

    
    trainer = Trainer(accelerator='cpu', max_epochs=50, logger=wand_logger, callbacks=[checkpoint_callback, stop_callback])
    trainer.fit(model, train_loader, val_loader)




if __name__ ==  '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--log_dir', type=str, default='/scratch/patel.purvi/Unet/logs')
    
    args = arg.parse_args()
    print(args.log_dir)
    
    main(args)