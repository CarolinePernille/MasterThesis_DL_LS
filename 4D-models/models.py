from conv4d import convNd

import torch.nn as nn
import torch

class UNet4D(nn.Module):
    def __init__(self):
        super(UNet4D, self).__init__()
        """ 
        A 4D-UNet model.
        """
        # Encoder block (each consisting of two conv layers with Group Norm and LeakyReLU as activation function)
        self.conv1 = self.double_conv(1,16)
        self.conv2 = self.double_conv(16,32)
        self.conv3 = self.double_conv(32,64)
        self.conv4 = self.double_conv(64,128)

        # Downsampling layers using strided convolutions
        self.downsample1 = convNd(in_channels = 16, out_channels = 16, num_dims = 4, kernel_size = 2, stride = (2,2,2,2), padding=0) 
        self.downsample2 = convNd(in_channels = 32, out_channels = 32, num_dims = 4, kernel_size = 2, stride = (2,2,2,2), padding=0) 
        self.downsample3 = convNd(in_channels = 64, out_channels = 64, num_dims = 4, kernel_size = 2, stride = (2,2,2,2), padding=0) 

        # Upsampling layers using transposed convolutions
        self.u1 = convNd(in_channels = 128, out_channels = 64, num_dims = 4, kernel_size = (5,4,4,4), stride = (2,2,2,2), padding=1, is_transposed=True)
        self.u2 = convNd(in_channels = 64, out_channels = 32, num_dims = 4, kernel_size = 4, stride = (2,2,2,2), padding=1, is_transposed=True)
        self.u3 = convNd(in_channels = 32, out_channels = 16, num_dims = 4, kernel_size = 4, stride = (2,2,2,2), padding=1, is_transposed=True)
  
        # Convolutional layers after concat
        self.up_conv1 = self.double_conv(128,64)
        self.up_conv2 = self.double_conv(64,32)
        self.up_conv3 = self.double_conv(32,16)

        # Final output layer
        self.out = convNd(16, 1, num_dims=4, kernel_size=1, stride=(1,1,1,1), padding=0)

    def double_conv(self, in_channels, out_channels):

        """ 
        A helper function to create two consecutive 4D convolutional layers with Group norm and Leaky ReLU as activation function.
        """
        conv = nn.Sequential(
                             convNd(in_channels, out_channels, num_dims=4, kernel_size=3, stride=(1,1,1,1), padding=1, use_bias=False),
                             nn.GroupNorm(num_groups = 8, num_channels=out_channels),
                             nn.LeakyReLU(inplace=True),
                             convNd(out_channels, out_channels, num_dims=4, kernel_size=3, stride=(1,1,1,1), padding=1, use_bias=False),
                             nn.GroupNorm(num_groups = 8, num_channels=out_channels),
                             nn.LeakyReLU(inplace=True)
                             )

        return conv

    def forward(self, x):
        """ 
        Forward pass through the 4D-UNet
        """
        down1 = self.conv1(x)
        down2 = self.downsample1(down1)
        down3 = self.conv2(down2)
        down4 = self.downsample2(down3)
        down5 = self.conv3(down4)
        down6 = self.downsample3(down5)
        down7 = self.conv4(down6)
        up1 = self.u1(down7)

        x = self.up_conv1(torch.cat([down5, up1], dim=1))
        up2 = self.u2(x)
        x = self.up_conv2(torch.cat([down3, up2], dim=1))
        up3 = self.u3(x)
        x = self.up_conv3(torch.cat([down1, up3],dim=1))
        out = self.out(x)
        return out
    

class complex_UNet4D(nn.Module):
    def __init__(self):
        super(complex_UNet4D, self).__init__()

        """ 
        A more complex version of 4D-UNet. Not in use for this current project. 
        """

        
        self.conv1 = self.double_conv(1,32)
        self.conv2 = self.double_conv(32,64)
        self.conv3 = self.double_conv(64,128)
        self.conv4 = self.double_conv(128,256)

        self.downsample1 = convNd(in_channels = 32, out_channels = 32, num_dims = 4, kernel_size = 2, stride = (2,2,2,2), padding=0) 
        self.downsample2 = convNd(in_channels = 64, out_channels = 64, num_dims = 4, kernel_size = 2, stride = (2,2,2,2), padding=0) 
        self.downsample3 = convNd(in_channels = 128, out_channels = 128, num_dims = 4, kernel_size = 2, stride = (2,2,2,2), padding=0) 
   
        self.u1 = convNd(in_channels = 256, out_channels = 128, num_dims = 4, kernel_size = (5,4,4,4), stride = (2,2,2,2), padding=1, is_transposed=True)
        self.u2 = convNd(in_channels = 128, out_channels = 64, num_dims = 4, kernel_size = 4, stride = (2,2,2,2), padding=1, is_transposed=True)
        self.u3 = convNd(in_channels = 64, out_channels = 32, num_dims = 4, kernel_size = 4, stride = (2,2,2,2), padding=1, is_transposed=True)


        self.up_conv1 = self.double_conv(256,128)
        self.up_conv2 = self.double_conv(128,64)
        self.up_conv3 = self.double_conv(64,32)

        self.out = convNd(32, 1, num_dims=4, kernel_size=1, stride=(1,1,1,1), padding=0)

    def double_conv(self, in_channels, out_channels):

        conv = nn.Sequential(
                             convNd(in_channels, out_channels, num_dims=4, kernel_size=3, stride=(1,1,1,1), padding=1, use_bias=True),
                             nn.ReLU(inplace=True),
                             convNd(out_channels, out_channels, num_dims=4, kernel_size=3, stride=(1,1,1,1), padding=1, use_bias=True),
                             nn.ReLU(inplace=True)
                             )

        return conv

    def forward(self, x):
        
        down1 = self.conv1(x)
        
        down2 = self.downsample1(down1)
        down3 = self.conv2(down2)
        down4 = self.downsample2(down3)
        down5 = self.conv3(down4)
        down6 = self.downsample3(down5)
        down7 = self.conv4(down6)
        up1 = self.u1(down7)

        x = self.up_conv1(torch.cat([down5, up1], dim=1))
        up2 = self.u2(x)
        x = self.up_conv2(torch.cat([down3, up2], dim=1))
        up3 = self.u3(x)
        x = self.up_conv3(torch.cat([down1, up3],dim=1))
        out = self.out(x)
        return out