# perceptual_loss.py

import torch
import torch.nn as nn
from torchvision.models import vgg16


class PerceptualLoss(nn.Module):
    def __init__(self, device="cuda"):
        """
        初始化感知损失模块
        使用预训练的 VGG16 提取特征，并计算 L1 损失
        :param device: 使用的设备，"cuda" 或 "cpu"
        """
        super(PerceptualLoss, self).__init__()
        # 加载 VGG16 的前16层（通常用于感知损失）
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)

        # 冻结参数，不进行训练
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        """
        前向传播，计算感知损失
        :param input: 生成图像，Tensor，形状 (B, C, H, W)
        :param target: 真实图像，Tensor，形状 (B, C, H, W)
        :return: 感知损失值
        """
        # 输入图像范围应为 [0, 1]，假设你已经做了 x.clamp(-1,1)*0.5+0.5
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return self.loss(input_features, target_features)
    

# perceptual_loss.py 对比学习感知损失

import torch
import torch.nn as nn
from torchvision.models import vgg16


class ContrastivePerceptualLoss(nn.Module):
    def __init__(self, device="cuda", n_patches=256, temperature=0.1, use_vgg=True):
        """
        对比学习风格的感知损失（PatchNCE 风格）
        
        Args:
            device: 使用的设备 ("cuda" 或 "cpu")
            n_patches: 每张图采样的 patch 数量
            temperature: InfoNCE 温度系数
            use_vgg: 是否使用 VGG 提取特征；否则可用 UNet 的 encoder
        """
        super(ContrastivePerceptualLoss, self).__init__()
        self.device = device
        self.n_patches = n_patches
        self.temperature = temperature
        
        if use_vgg:
            # 使用 VGG16 前若干层作为特征提取器（通常取 relu3_1 或 relu2_2）
            vgg = vgg16(pretrained=True).features[:16].eval().to(device)  # relu2_2
            for param in vgg.parameters():
                param.requires_grad = False
            self.feature_extractor = vgg
        else:
            self.feature_extractor = None  # 可替换为你的模型 encoder
        
        self.criterion = nn.CrossEntropyLoss()

    def _extract_patches(self, feat_map, patch_size=8):
        """
        从特征图中随机采样空间 patch
        Args:
            feat_map: 特征图 [B, C, H, W]
            patch_size: 每个 patch 的大小
        Returns:
            patches: [B, n_patches, C*patch_size*patch_size]
        """
        B, C, H, W = feat_map.shape
        patches = []
        for _ in range(self.n_patches):
            i = torch.randint(0, H - patch_size + 1, (B,))
            j = torch.randint(0, W - patch_size + 1, (B,))
            patch = torch.stack([
                feat_map[b, :, i[b]:i[b]+patch_size, j[b]:j[b]+patch_size] 
                for b in range(B)
            ])  # [B, C, P, P]
            patch = patch.reshape(B, C, -1).mean(dim=-1)  # Global Average Pooling → [B, C]
            patches.append(patch)
        # 拼接所有 patch → [B, n_patches, C]
        return torch.stack(patches, dim=1)

    def forward(self, input, target):
        """
        计算对比感知损失
        Args:
            input: 生成图像 [B, C, H, W]，范围 [0,1]
            target: 真实图像 [B, C, H, W]，范围 [0,1]
        Returns:
            loss: 标量，对比损失值
        """
        B = input.size(0)

        with torch.no_grad():
            target_feats = self.feature_extractor(target)  # [B, C, H', W']
        input_feats = self.feature_extractor(input)  # [B, C, H', W']

        # 提取 patch 特征
        input_patches = self._extract_patches(input_feats)   # [B, n_patches, C]
        target_patches = self._extract_patches(target_feats) # [B, n_patches, C]

        # 归一化特征（L2 norm）
        input_patches = torch.nn.functional.normalize(input_patches, p=2, dim=-1)  # [B, N, C]
        target_patches = torch.nn.functional.normalize(target_patches, p=2, dim=-1)  # [B, N, C]

        total_loss = 0.0
        for b in range(B):
            # 正样本：相同位置的 patch 对 (i,i)
            pos_sim = torch.sum(input_patches[b] * target_patches[b], dim=-1, keepdim=True) / self.temperature  # [N, 1]

            # 负样本：input[b,i] vs target[b,j≠i]
            neg_sim = torch.matmul(input_patches[b], target_patches[b].t()) / self.temperature  # [N, N]
            mask = torch.eye(self.n_patches, dtype=torch.bool, device=self.device)
            neg_sim = neg_sim.masked_fill(mask, -1e9)  # 排除对角线

            # 拼接正负样本 logits
            logits = torch.cat([pos_sim, neg_sim], dim=1)  # [N, 1 + N]
            labels = torch.zeros(self.n_patches, dtype=torch.long, device=self.device)  # 第0列是正样本

            total_loss += self.criterion(logits, labels)

        return total_loss / B