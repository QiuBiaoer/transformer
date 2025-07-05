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




class Attention(nn.Module):
    # dim输入的token维度768, num_heads注意力头数，qkv_bias生成QKV的时候是否添加偏置，
    # qk_scale用于缩放QK的缩放因子，若为None，则使用1/sqrt(embed_dim_pre_head)
    # atte_drop_ration注意力分数的dropout的比率，防止过拟合  proj_drop_ration最终投影层的dropout的比率
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, atte_drop_ration=0., proj_drop_ration=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # qk的缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 通过全连接层生成QKV，为了并行运算提高计算效率，同时参数更少
        self.attn_drop = nn.Dropout(atte_drop_ration)
        self.proj_drop = nn.Dropout(proj_drop_ration)
        # 将每个head得到的输出进行concat拼接，然后通过线性变换映射回原本的嵌入dim
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x):
        B, N, C = x.shape  # B为batch,N为num_patch+1,C为embed_dim  +1为clstoken
        #  B N 3*C -> B N 3 num_heads, C//self.num_heads -> 3 B num_heads N C//self.num_heads  作用是方便之后的运算
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 用切片拿到QKV,形状是 B num_heads N C//self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算qk的点积，并进行缩放得到注意力分数
        # Q: [3 B num_heads N C//self.num_heads] k.transpose(-2,-1)  K:[B num_heads C//self.num_heads N]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B num_heads N N
        attn = attn.softmax(dim=-1) # 对每行进行处理 使得每行的和为1
        # 注意力权重对V进行加权求和
        # attn @ v : B num_heads N C//self.num_heads
        # transpose: B N self.num_heads C//self.num_heads
        # reshape将最后两个维度拼接，合并多个头的输出，回到总的嵌入维度
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()


