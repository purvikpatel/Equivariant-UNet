from torch import nn, optim, utils, Tensor
from torch.nn import functional as F

import lightning.pytorch as pl

import torch

class Unet(pl.LightningModule):
    def __init__(self, num_classes, bilinear=True, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.bilinear = bilinear
        self.num_classes = num_classes
        self.dropout = dropout
        def conv_block(in_channels, out_channels, dropout=self.dropout):
            block =  nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            if dropout:
                block.add_module('dropout', nn.Dropout2d(dropout))
            return block
                
        
        def down_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                conv_block(in_channels, out_channels)
            )

        class up_conv(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()
                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, 2, stride=2)

                self.conv = conv_block(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)


        self.conv1 = conv_block(3, 64)
        self.down1 = down_conv(64, 128)
        self.down2 = down_conv(128, 256)
        self.down3 = down_conv(256, 512)
        self.down4 = down_conv(512, 512)

        self.up1 = up_conv(1024, 256, self.bilinear)
        self.up2 = up_conv(512, 128, self.bilinear)
        self.up3 = up_conv(256, 64, self.bilinear)
        self.up4 = up_conv(128, 64, self.bilinear)
        self.out = nn.Conv2d(64, self.num_classes, 1)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y = y - 1
        y =y.squeeze(1)
        y = y.long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y -1
        y =y.squeeze(1)
        y = y.long()
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)
    
    
    


if __name__ == '__main__':
    model = Unet(3, bilinear=False, dropout=0)
    print(model)
    x = torch.randn(1, 3, 240, 240)
    print(model(x).shape)


        