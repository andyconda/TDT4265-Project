import torch
from torch import nn
import torchvision
#from ssd.data.transforms.target_transform import SSDTargetTransform
#from ssd.data.transforms.transforms import *

#print(torchvision.models.resnet34(pretrained=True))

class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[4], 3, 3),
     shape(-1, output_channels[5], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
     
        self.model = torchvision.models.resnet34(pretrained=True)
        for param in self.model.parameters():  # Freeze all parameters
            param.requires_grad = False
        for param in self.model.layer4.parameters():  # Unfreeze the last 5 convolutional
            param.requires_grad = True  # layers
    
        self.bank0 = torch.nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu)
        self.bank1 = torch.nn.Sequential(
                self.model.maxpool,
                self.model.layer1)
        self.bank2 = self.model.layer2
        self.bank3 = self.model.layer3
        self.bank4 = self.model.layer4
        self.bank5 = self.model.avgpool
    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[4], 3, 3),
            shape(-1, output_channels[5], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        
        out_features = [None] * 6
        out_features[0] = self.bank0(x).cuda()
        out_features[1] = self.bank1(out_features[0]).cuda()
        out_features[2] = self.bank2(out_features[1]).cuda()
        out_features[3] = self.bank3(out_features[2]).cuda()
        out_features[4] = self.bank4(out_features[3]).cuda()
        out_features[5] = self.bank5(out_features[4]).cuda()

        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
