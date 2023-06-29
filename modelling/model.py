from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

# -------------------------------------------------
class SiameseNetwork_SV(nn.Module):
    def __init__(self, backbone="resnet50"):
        """
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        """

        super().__init__()

        if backbone == "resnet50":
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.backbone_module = nn.Sequential(
            *[
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ]
        )

        for name, param in self.backbone_module.named_parameters():
            param.requires_grad = False

        self.ada_avg = nn.AdaptiveAvgPool2d((1, 1))

        conv_1x1_1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1)
        conv_1x1_2 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        conv_1x1_3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        bn_1 = torch.nn.BatchNorm2d(512)
        bn_2 = torch.nn.BatchNorm2d(64)
        relu = torch.nn.LeakyReLU(negative_slope=0.05)
        sig = torch.nn.Sigmoid()

        self.classification_module = nn.Sequential(
            *[conv_1x1_1, bn_1, relu, conv_1x1_2, bn_2, relu, conv_1x1_3, sig]
        )

    def forward_once(self, x):
        x = self.backbone_module(x)
        x = self.ada_avg(x)
        return x

    def forward(self, img1, img2):
        x1 = self.forward_once(img1)
        x2 = self.forward_once(img2)
        x = x1 - x2
        o = self.classification_module(x)
        o = o.squeeze(-1).squeeze(-1)
        return o
