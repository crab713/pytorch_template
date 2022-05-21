import torch
from torch import Tensor
import cv2
from torch.nn import functional as F
import torch.nn as nn

from .backbone.swin import SwinTransformer
from .head.upernet import UPerHead


class Elimator(nn.Module):
    def __init__(self, img_size = 512, in_channel = 3, num_classes = 3, embed_dim = 96):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.backbone = SwinTransformer(pretrain_img_size=img_size, in_chans=in_channel, embed_dim=embed_dim)
        self.decoder = UPerHead(in_channels=[embed_dim, embed_dim*2, embed_dim*4, embed_dim*8],
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=512,
                                dropout_ratio=0.1,
                                num_classes=num_classes)
        self.UpSample = nn.Upsample((img_size, img_size), mode='bilinear', align_corners=False)

        self.backbone.init_weights()
    
    def check_input(self, inputs: Tensor) -> None:
        if not isinstance(inputs, Tensor):
            raise RuntimeError(
                "Input has inconsistent type: got {}, excepted torch.Tensor".format(
                    type(inputs)))
        
        if inputs.size(2) != self.img_size or inputs.size(3) != self.img_size:
            raise RuntimeError(
                "Input has inconsistent image shape: got {}, except {}".format(
                    (inputs.size(2), inputs.size(3)), self.img_size))

    def forward(self, inputs: Tensor) -> Tensor:
        self.check_input(inputs)
        features = self.backbone(inputs)
        seg_logit = self.decoder(features)
        output = self.UpSample(seg_logit)
        return output