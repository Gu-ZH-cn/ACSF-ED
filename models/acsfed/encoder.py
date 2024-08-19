import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
from ..basic.conv import Conv2d

class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1, block_size = [4,2,1],halo_size = [4,4,4]):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)
        self.reset_parameters()
        # SLSA
        self.block_size = block_size
        self.halo_size = halo_size
        self.rel_pos_emb1 = RelPosEmb(
            block_size = block_size[0],
            rel_size = block_size[0] + (halo_size[0] * 2),
            dim_head = self.head_dim
        )
        self.rel_pos_emb2 = RelPosEmb(
            block_size = block_size[1],
            rel_size = block_size[1] + (halo_size[1] * 2),
            dim_head = self.head_dim
        )
        self.rel_pos_emb3 = RelPosEmb(
            block_size = block_size[2],
            rel_size = block_size[2] + (halo_size[2] * 2),
            dim_head = self.head_dim
        )

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w, = q.shape
        device = q.device
        h_out, w_out = h // self.stride, w // self.stride

        ### att
        '''
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)
        '''
        if (h == 28):
            block, halo, heads = self.block_size[0], self.halo_size[0], self.head
        elif (h == 14):
            block, halo, heads = self.block_size[1], self.halo_size[1], self.head
        else:
            block, halo, heads = self.block_size[2], self.halo_size[2], self.head
        # block, halo, heads = self.block_size, self.halo_size, self.head
        q_att = rearrange(q, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=block, p2=block)
        k_att = F.unfold(k, kernel_size=block + halo * 2, stride=block, padding=halo)
        v_att = F.unfold(v, kernel_size=block + halo * 2, stride=block, padding=halo)
        k_att = rearrange(k_att, 'b (c j) i -> (b i) j c', c=c)
        v_att = rearrange(v_att, 'b (c j) i -> (b i) j c', c=c)
            # split heads(从通道划分检测头)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q_att, k_att, v_att))
        q1 *= scaling
            # attention
        sim = einsum('b i d, b j d -> b i j', q1, k1)
            # add relative positional bias
        if(h==28):
            sim += self.rel_pos_emb1(q1)
        elif(h==14):
            sim += self.rel_pos_emb2(q1)
        else:
            sim += self.rel_pos_emb3(q1)
            # mask out padding (in the paper, they claim to not need masks, but what about padding?)
        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block + (halo * 2), stride=block, padding=halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b=b, h=heads)
        mask = mask.bool()
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)
            # attention
        attn = sim.softmax(dim=-1)
            # aggregate
        out = einsum('b i j, b j d -> b i d', attn, v1)
            # merge and combine heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        out_att = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=(h // block), w=(w // block),
                                p1=block, p2=block)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

# Channel Self Attetion Module
class CSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


# Spatial Self Attetion Module
class SSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(SSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1).permute(0, 2, 1)   # [B, N, C]
        key = x.view(B, C, -1)                      # [B, C, N]
        value = x.view(B, C, -1).permute(0, 2, 1)   # [B, N, C]

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


# Channel Encoder
class ChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            # ACmix(in_dim,out_dim),
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            CSAM(),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x1, x2):
        """
            x: [B, C, H, W]
        """
        x = torch.cat([x1, x2], dim=1)
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


# Spatial Encoder
class SpatialEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SSAM(),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        """
            x: [B, C, H, W]
        """
        x = self.fuse_convs(x)

        return x

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

def build_channel_encoder(cfg, in_dim, out_dim):
    encoder = ChannelEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm']
        )

    return encoder

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)
def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def to(x):
    return {'device': x.device, 'dtype': x.dtype}