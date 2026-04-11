import pytest
import torch
import torch.nn as nn
from models.backbone import ResNet50Backbone
from models.bottleneck import Bottleneck
from models.arcface import ArcFace


class TestResNet50Backbone:
    def test_backbone_output_shape(self):
        """验证 backbone 输出形状为 (B, 2048)"""
        model = ResNet50Backbone(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 2048), f"Expected (2, 2048), got {output.shape}"
    
    def test_backbone_freezing(self):
        """验证 layer1-3 被冻结，layer4 可训练"""
        model = ResNet50Backbone(pretrained=False, freeze_until_layer=3)
        
        # layer1-3 应该被冻结
        assert not model.layer1[0].conv1.weight.requires_grad
        assert not model.layer2[0].conv1.weight.requires_grad
        assert not model.layer3[0].conv1.weight.requires_grad
        
        # layer4 应该可训练
        assert model.layer4[0].conv1.weight.requires_grad
    
    def test_backbone_unfreeze(self):
        """验证 unfreeze_all() 方法"""
        model = ResNet50Backbone(pretrained=False, freeze_until_layer=3)
        model.unfreeze_all()
        
        # 所有参数都应该可训练
        for param in model.parameters():
            assert param.requires_grad, "unfreeze_all() 后所有参数应该 requires_grad=True"


class TestBottleneck:
    def test_bottleneck_output_shape(self):
        """验证 bottleneck 输出形状为 (B, 512)"""
        model = Bottleneck(in_features=2048, bottleneck_dim=512)
        x = torch.randn(4, 2048)
        output = model(x)
        assert output.shape == (4, 512), f"Expected (4, 512), got {output.shape}"
    
    def test_bottleneck_l2_normalization(self):
        """验证输出 L2 范数为 1"""
        model = Bottleneck(in_features=2048, bottleneck_dim=512)
        x = torch.randn(4, 2048)
        output = model(x)
        
        # 计算每个样本的 L2 范数
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"L2 norms should be 1, got {norms}"


class TestArcFace:
    def test_arcface_output_shape(self):
        """验证 ArcFace 输出形状为 (B, num_classes)"""
        model = ArcFace(in_features=512, num_classes=8)
        x = torch.randn(4, 512)
        label = torch.tensor([0, 1, 2, 3])
        output = model(x, label)
        assert output.shape == (4, 8), f"Expected (4, 8), got {output.shape}"


class TestFullModel:
    def test_full_model_forward(self):
        """验证完整模型 forward: backbone → bottleneck → arcface"""
        backbone = ResNet50Backbone(pretrained=False)
        bottleneck = Bottleneck(in_features=2048, bottleneck_dim=512)
        arcface = ArcFace(in_features=512, num_classes=8)
        
        backbone.eval()
        bottleneck.eval()
        arcface.eval()
        
        x = torch.randn(2, 3, 224, 224)
        label = torch.tensor([0, 1])
        
        with torch.no_grad():
            features = backbone(x)
            features = bottleneck(features)
            output = arcface(features, label)
        
        assert output.shape == (2, 8), f"Expected (2, 8), got {output.shape}"
        assert features.shape == (2, 512), f"Expected features (2, 512), got {features.shape}"
