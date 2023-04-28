import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from model import Unet
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import seed_everything
from utils import compute_metrics


def main(args):
    
    model = Unet.load_from_checkpoint(args.model_path)
    model.eval()
    
    image_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
    ])
    dataset = OxfordIIITPet(root='../../data',download=True, target_types='segmentation',
                                     transform=image_transform, target_transform=target_transform)

    
    val = 100
    train = len(dataset) - val
    train_set, val_set = random_split(dataset, [train, val])

    dataloader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)

    ious_class1 = []
    ious_class2 = []
    ious_class3 = []
    dices_class1 = []
    dices_class2 = []
    dices_class3 = []

    for i, (x, y) in enumerate(dataloader):
        print(f'Batch {i+1}')
        y = y.squeeze(1)
        print(f'y shape : {y.shape}')
        output = model(x)
        print(f'output shape : {output.shape}')
        ious, dices = compute_metrics(output, y)
        ious_class1.append(ious[0])
        ious_class2.append(ious[1])
        ious_class3.append(ious[2])
        dices_class1.append(dices[0])
        dices_class2.append(dices[1])
        dices_class3.append(dices[2])

    iou_1 = torch.stack(ious_class1).mean()
    iou_2 = torch.stack(ious_class2).mean()
    iou_3 = torch.stack(ious_class3).mean()
    dice_1 = torch.stack(dices_class1).mean()
    dice_2 = torch.stack(dices_class2).mean()
    dice_3 = torch.stack(dices_class3).mean()

    print(f'IOU class 1: {iou_1}')
    print(f'IOU class 2: {iou_2}')
    print(f'IOU class 3: {iou_3}')
    print(f'DICE class 1: {dice_1}')
    print(f'DICE class 2: {dice_2}')
    print(f'DICE class 3: {dice_3}')
    
    
    # output = F.softmax(output, dim=1)
    # output = torch.argmax(output, dim=1)
    # output = output.squeeze().detach().cpu().numpy()


    # fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    # ax[0].imshow(dataset[123][0].permute(1, 2, 0))
    # ax[1].imshow(dataset[123][1].permute(1, 2, 0))
    # ax[2].imshow(output)
    # plt.show()




if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--model_path', type=str, required=True, help='Path to the model')

    args = arg.parse_args()
    seed_everything(42)
    main(args)