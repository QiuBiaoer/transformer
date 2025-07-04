import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        # img_size图像大小   patch_size每个图像块patch的大小  in_c 输入通道  embed_dim 嵌入维度  norm_layer 可选的归一化层
        super().__init__()
        img_size = (img_size, img_size)   # 将输入图像大小变为二维元组
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16, 224/16  以patch为单位形成的新“图像”尺寸
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 14*14=196 patch总数

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size) # 利用一个卷积核为16*16，步长为16大小进行卷积操作来等效实现将原图拆分成patch   B, 3, 224, 224 -> B, 768, 14, 14
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 若存在norm layer则使用，否则保持不变

    def forward(self, x):
        B, C, H, W = x.shape   # 获取输入张量的形状
        assert H == self.img_size[0] and W == self.img_size[1],\
        f"输入图像大小{H} * {W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        # B, 3, 224, 224 -> B, 768, 14, 14 -> B, 768, 196 -> B, 196, 768
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x) # 使用norm层进行归一化
        return x