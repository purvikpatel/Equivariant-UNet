from torch import nn, optim, utils, Tensor
from torch.nn import functional as F

import lightning.pytorch as pl
import torchmetrics.functional as metrics

import torch
from utils import *

from escnn import nn as escnn_nn
from escnn import gspaces



class EquivariantUnet(pl.LightningModule):
    def __init__(self, num_classes, N=4, reflections=False, bilinear=True, dropout=0.2, dropout_type='Field'):
        super().__init__()
        self.save_hyperparameters()

        self.bilinear = bilinear
        self.num_classes = num_classes
        self.dropout = dropout
        self.ious = []
        self.dice_scores = []
        self.N = N

        if reflections and N == 1:
            self.r2_act = gspaces.flip2dOnR2()
        
        elif reflections:
            self.r2_act = gspaces.flipRot2dOnR2(N=N)
        
        else:
            self.r2_act = gspaces.rot2dOnR2(N=N)


        in_type = escnn_nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])  # 3 channels
		
        self.input_type = in_type

        def eq_conv_block(in_type, out_type, dropout=self.dropout, dropout_type='Field'):
            block = escnn_nn.SequentialModule(
				escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
				escnn_nn.InnerBatchNorm(out_type),
				escnn_nn.ReLU(out_type, inplace=True),
				escnn_nn.R2Conv(out_type, out_type, kernel_size=3, padding=1),
				escnn_nn.InnerBatchNorm(out_type),
				escnn_nn.ReLU(out_type, inplace=True),
            )
            if dropout:
                if dropout_type == 'Field':
                    block.add_module("dropout", escnn_nn.FieldDropout(out_type, dropout))
                elif dropout_type == 'Point':
                    block.add_module("dropout", escnn_nn.PointwiseDropout(out_type, dropout))
            return block

        def conv_block(in_channels, out_channels, dropout=self.dropout):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            if dropout:
                block.add_module("dropout", nn.Dropout2d(dropout))
            return block

        def down_conv(in_type, out_type):
            return escnn_nn.SequentialModule(escnn_nn.PointwiseMaxPool(in_type, 2, padding=1), eq_conv_block(in_type, out_type))

        class up_conv(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()
                if bilinear:
                    self.up = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    )
                else:
                    self.up = nn.ConvTranspose2d(
                        in_channels // 2, in_channels // 2, 2, stride=2
                    )

                self.conv = conv_block(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        
        x = self.r2_act.regular_repr.size #size of the regular representation 

        out_type = escnn_nn.FieldType(self.r2_act, (64 // x)*[self.r2_act.regular_repr]) 
        
        self.conv1 = eq_conv_block(in_type, out_type)
        in_type = out_type
        out_type = escnn_nn.FieldType(self.r2_act, (128 // x)*[self.r2_act.regular_repr])
        self.down1 = down_conv(in_type, out_type)

        in_type = out_type
        out_type = escnn_nn.FieldType(self.r2_act, (256 // x)*[self.r2_act.regular_repr])
        self.down2 = down_conv(in_type, out_type)
        self.down2_out = out_type

        in_type = out_type
        out_type = escnn_nn.FieldType(self.r2_act, (512 // x)*[self.r2_act.regular_repr])
        self.down3 = down_conv(in_type, out_type)

        in_type = out_type
        out_type = escnn_nn.FieldType(self.r2_act, (512 // x)*[self.r2_act.regular_repr])
        self.down4 = down_conv(in_type, out_type)

        # self.down1 = down_conv(64, 128)
        # self.down2 = down_conv(128, 256)
        # self.down3 = down_conv(256, 512)
        # self.down4 = down_conv(512, 512)

     

        self.up1 = up_conv(1024, 256, self.bilinear)
        self.up2 = up_conv(512, 128, self.bilinear)
        self.up3 = up_conv(256, 64, self.bilinear)
        self.up4 = up_conv(128, 64, self.bilinear)
        self.out = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        x = escnn_nn.GeometricTensor(x, self.input_type)
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        x = self.up1(x5.tensor, x4.tensor)
        x = self.up2(x, x3.tensor)
        x = self.up3(x, x2.tensor)
        x = self.up4(x, x1.tensor)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y - 1
        y = y.squeeze(1)
        y = y.long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y - 1
        y = y.squeeze(1)
        y = y.long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        iou = metrics.classification.multiclass_jaccard_index(
            y_hat, y, num_classes=3, ignore_index=2, average="weighted"
        )
        dice_score = metrics.classification.dice(
            y_hat, y, num_classes=3, ignore_index=2, average="weighted"
        )

        self.ious.append(iou)
        self.dice_scores.append(dice_score)

        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        iou = torch.stack(self.ious).mean()
        dice = torch.stack(self.dice_scores).mean()
        self.log("val_iou", iou, on_step=False, on_epoch=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    model = Unet(3, N=8, bilinear=False, dropout=0.3, reflections=True)
    #print(model) 
    model.train()
    x = torch.randn(8, 3, 241, 241)
    print(model(x).shape)
