import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True, freeze_until_layer=3):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        if freeze_until_layer > 0:
            self._freeze_layers(freeze_until_layer)
    
    def _freeze_layers(self, freeze_until_layer):
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers[:freeze_until_layer]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
