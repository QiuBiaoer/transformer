from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from timm.layers import DropPath


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
        # in_features输入的维度, hidden_features隐藏层维度、通常为in_features的4倍, out_features输出维度、通常与输入维度相等
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class Block(nn.Module):
    # mlp_ratio 计算hidden_features大小 默认为输入4倍   norm_layer正则化层
    # drop_path_ratio 是drop_path的比率，该操作在残差连接之前  drop_ratio 是多头自注意力机制最后的linear后使用的dropout

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # transformer encoder block中的第一个layer norm
        # 实例化多头注意力机制
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              atte_drop_ration=attn_drop_ratio, proj_drop_ration=drop_path_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 计算MLP第一个全连接层的节点数
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12,mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0. , embed_layer=PatchEmbed ,norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        # 设置一个较小的参数防止除0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU()
        self.patch_embed = embed_layer(img_size, patch_size, in_c, embed_dim, norm_layer)
        num_patches = self.patch_embed.num_patches  # 得到patches的个数
        # 使用nn.Parameter构建可训练的参数，用零矩阵初始化，第一个为batch，后两个为1*768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # pos_embed 大小与concat拼接后的大小一致，是197*768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)
        # 根据传入的drop_path_ratio 构建等差序列，从0到drop_path_ratio，有depth个元素
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # 使用nn.Sequential将列表中的所有模块打包为一个整体 depth对应的是使用了transformer encoder block的数量
        self.block = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)  # 通过transformer后的layer norm
        '''
            这段代码中logits层是作为模型最后一层的原始输出值（一般是全连接层，尚未经过归一化），一般需要通过激活函数得到统计概率作为最终输出
            这里的representation size指的是你想要的输出数据的尺寸大小  在小规模的ViT中不需要该参数
        '''
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity() # 不做任何处理
        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):  # 针对patch embedding部分的forward
        # B C H W -> B num_patches embed_dim  196 * 768
        x = self.patch_embed(x)
        # 1, 1, 768 -> B, 1, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # dist_token存在， 则拼接dist_token和cls_token, 否则只拼接cls_token和输入的patch特征x
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1) # B 197 768
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1),x), dim=1)

        x = self.pos_drop(x+self.pos_embed)
        x = self.block(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])  # dist_token为None，利用切片的形式获取cls_token对应的输出
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            # 知识蒸馏相关知识
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            # 如果是训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                # 则返回两个头部的预测结果
                return x, x_dist
        else:
            x = self.head(x) # 最后的linear全连接层
        return x


def _init_vit_weights(m):
    # 判断模块m是否为线形层
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None: # 如果线性层存在偏置项，则将偏置项初始化为0
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out') # 对卷积层的权重做一个初始化，适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  # 对层归一化的权重初始化为1


def vit_base_patch16_224(num_classes:int = 1000, pretrained=False):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                              representation_size=None, num_classes=num_classes)
    return model


