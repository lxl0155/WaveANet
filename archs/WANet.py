import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision import ops
from ..utils.registry import ARCH_REGISTRY
# from basicsr.utils.registry import ARCH_REGISTRY

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class BSConvU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

#前
class A(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = conv_layer(in_channels, in_channels, 1)
        self.dc = self.distilled_channels = in_channels//2 #18
        self.rc = self.remaining_channels = in_channels #36
        self.c1_d = conv_layer(in_channels, self.dc, 1) #18
        self.c1_r = BSConvU(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros") #36
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1) #18
        self.c2_r = BSConvU(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros") #36
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1) #18
        self.c3_r = BSConvU(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros") #36
        self.c4 = BSConvU(self.remaining_channels, self.dc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros") #18
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)


    def forward(self, input):
        input = self.conv(input)
        input_copy = input

        distilled_c1 = self.c1_d(input) #16
        r_c1 = self.act(self.c1_r(input)) #32

        distilled_c2 = self.c2_d(r_c1) #16
        r_c2 = self.act(self.c2_r(r_c1)) #32


        distilled_c3 = self.c3_d(r_c2) #16
        r_c3 = self.act(self.c3_r(r_c2)) #32

        r_c4 = self.act(self.c4(r_c3)) #16

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.act(self.c5(out)) + input_copy

        return out_fused


#后
class RowAttention(nn.Module):
    def __init__(self, dim, bias):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(RowAttention, self).__init__()
        self.dim = dim

        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(y)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)


        # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        row_attn = torch.bmm(Q, K)
        ########
        # 此时的 row_atten的[:,i,0:w] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有行的 所有列(0:w)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
        ########

        # 对row_attn进行softmax
        row_attn = row_attn.softmax(dim=2)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
        out = torch.bmm(V, row_attn.permute(0, 2, 1))

        # size = (b,c1,h,2)
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)

        out = self.gamma * out + x

        return out


class ColAttention(nn.Module):
    def __init__(self, dim, bias):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.in_dim = dim

        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x, y):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(y)  # size = (b, c1,h,w)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)


        # size = (b*w,h,h) [:,i,j] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的第 Hj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        col_attn = torch.bmm(Q, K)
        ########
        # 此时的 col_atten的[:,i,0:w] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的 所有列(0:h)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:h）逐个位置的值的乘积，得到列attn
        ########

        # 对row_attn进行softmax
        col_attn = col_attn.softmax(dim=2)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*w,c1,h) 这里先需要对col_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 col_attn的行的乘积，即求权重和
        out = torch.bmm(V, col_attn.permute(0, 2, 1))

        # size = (b,c1,h,w)
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)

        out = self.gamma * out + x
        return out



class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode='channel'):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q(x)  # image
        k = self.k(x)  # event
        v = self.v(x)  # event

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class B(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.GELU()
        self.conv_1 = BSConvU(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros")
        self.conv_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv = nn.Conv2d(dim*3, dim, 1, 1, 0)
        self.att = SelfAttention(dim, num_heads=2, bias=False, mode='channel')
        self.row = RowAttention(dim, bias=False)
        self.col = ColAttention(dim, bias=False)

    def forward(self, x):
        x_1 = self.conv_2(self.act(self.conv_1(x)))
        x_1 += self.conv_0(x)
        out_1 = self.att(x_1)
        out_2 = self.row(x_1, out_1)
        out_3 = self.col(x_1, out_1)
        out = torch.cat([out_1, out_2, out_3], dim=1)
        out = self.act(self.conv(out))

        return out



class Block(nn.Module):
    def __init__(self, dim=36, ffn_scale=2):
        super(Block, self).__init__()
        self.a = A(dim)
        self.b = B(dim)
        self.pixel_norm = nn.LayerNorm(dim * ffn_scale)

        self.remaining_channels = dim // ffn_scale
        self.other_channels = dim - self.remaining_channels
        self.conv_0 = nn.Conv2d(dim * ffn_scale, dim, 1, 1, 0)
        self.conv_1 = BSConvU(self.remaining_channels, self.remaining_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros")
        self.conv_2 = nn.Conv2d(self.other_channels, self.other_channels, 1, 1, 0)
        self.conv_3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gelu = nn.GELU()




    def forward(self, f):
        a = self.a(f)
        b = self.b(f)
        fea = torch.cat((a, b), 1)

        fea = fea.permute(0, 2, 3, 1)
        fea = self.pixel_norm(fea)
        fea = fea.permute(0, 3, 1, 2).contiguous()
        fea = self.conv_0(fea)
        fea1, fea2 = torch.split(fea, [self.remaining_channels, self.other_channels], dim=1)
        fea1 = self.gelu(self.conv_1(fea1))
        fea2 = self.gelu(self.conv_2(fea2))
        out = torch.cat((fea1, fea2), 1)
        out = self.gelu(self.conv_3(out)) + fea
        return out





#整体
@ARCH_REGISTRY.register()
class WANet(nn.Module):
    def __init__(self, dim=36, n_blocks=8, ffn_scale=2, upscaling_factor=2):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.B1 = Block(dim, ffn_scale)
        self.B2 = Block(dim, ffn_scale)
        self.B3 = Block(dim, ffn_scale)
        self.B4 = Block(dim, ffn_scale)
        self.B5 = Block(dim, ffn_scale)
        self.B6 = Block(dim, ffn_scale)
        self.B7 = Block(dim, ffn_scale)
        self.B8 = Block(dim, ffn_scale)


        self.c1 = nn.Conv2d(dim * n_blocks, dim, 1, 1, 0)
        self.GELU = nn.GELU()

        self.to_img = nn.Sequential(
            BSConvU(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        out_B1 = self.B1(x)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)


        trunk = torch.cat(
            [out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1
        )
        out_B = self.GELU(self.c1(trunk)) + x

        out = self.to_img(out_B)
        return out
