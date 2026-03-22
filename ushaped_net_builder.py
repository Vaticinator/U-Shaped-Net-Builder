import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.utils import save_image

class UShapedNet(nn.Module):

    
    def save_x(self, x, label="layer", wait=False):
        # save all features of the first image in batch
        save_image(x[0].view(-1,1,x.shape[2],x.shape[3]), "{}.png".format(label), nrow=int(x.shape[1]**0.5), normalize=True)
        if wait: input("Features saved. Press any key to continue.")

    def __init__(   self,
                    ch_in, ch_out, 
                    init_features = 64, 
                    u_blocks_amount =  [ 1,  2,  2,  6,  2,  2,  1 ], 
                    u_blocks_variant = ['C','C','C','R','C','C','C'], 
                    u_blocks_resize =  ['N','D','D','N','U','U','N'], 
                    u_connected = True,
                    use_dropout = False,
                    padding_mode = 'reflect',
                    fin_act = nn.Tanh()
        ):
        
        assert(len(u_blocks_amount) == len(u_blocks_variant) == len(u_blocks_resize))
        super(UShapedNet, self).__init__()
        self.description = f'--------unet--------\nINIT Conv {ch_in} -> {init_features}\n'
        
        mid_features = init_features // 2 if init_features >= 16 else init_features
        self.initial = nn.Sequential(
            # 1st (3x3)
            nn.Conv2d(ch_in, mid_features, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.GELU(),            
            # 3nd (3x3)
            nn.Conv2d(mid_features, init_features, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(init_features),
            nn.GELU()
        )

        self.u_connected = u_connected
        if self.u_connected:
            u_blocks_resize_np = np.array(u_blocks_resize, dtype=str)
            self.connect_d = np.where(u_blocks_resize_np=='D')[0] #[1,2]
            self.connect_u = np.where(u_blocks_resize_np=='U')[0] #[4,5]
        
        features_in = init_features
        features_out = init_features
        self.network = nn.ModuleList([])
        self.skip_adapters = nn.ModuleDict()
        for i in range(len(u_blocks_amount)):
            
            sampling = "up" if u_blocks_resize[i] == 'U' else "down" if u_blocks_resize[i] == 'D' else None
            features_out = features_out//2 if u_blocks_resize[i] == 'U' else features_out*2 if u_blocks_resize[i] == 'D' else features_out            
            self.description += f'{u_blocks_amount[i]}, {u_blocks_variant[i]}, {u_blocks_resize[i]}, {features_in} -> {features_out}\n'
            
            seq_list = []
            for n in range(u_blocks_amount[i]):
                
                if u_blocks_variant[i] == 'C':
                    seq_list.append(ConvBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'R':
                    seq_list.append(ResidualBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'SC':
                    seq_list.append(SeparableConvBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'SR':
                    seq_list.append(SeparableResidualBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'SE':
                    seq_list.append(SEResidualBlock(features_in, features_out, sampling=sampling, use_dropout=use_dropout, padding_mode=padding_mode))
                elif u_blocks_variant[i] == 'DA':
                    seq_list.append(DualAttentionBlock(features_in))
                elif u_blocks_variant[i] == 'XA':
                    seq_list.append(CrissCrossAttention(features_in))
                
                features_in = features_out
                sampling = None
            self.network.append(nn.Sequential(*seq_list))
            
            # u_connections adapters
            if self.u_connected and i in self.connect_u:
                self.skip_adapters[str(i)] = nn.Sequential(
                    nn.Conv2d(features_out * 2, features_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(features_out),
                    nn.GELU()
                )
        
        self.description += f'FINAL Conv {features_in} -> {ch_out}\nFINAL Act {fin_act}\n--------------------'
        self.final = nn.Sequential(
            nn.Conv2d(features_out, ch_out, kernel_size=7, padding=3, padding_mode=padding_mode),
            fin_act
        )
        
    def forward(self, x, debug=False):
        h =[]
        x = self.initial(x)
        for i in range(len(self.network)):
            if self.u_connected and i in self.connect_d: h.append(x)
            x = self.network[i](x)
            if self.u_connected and i in self.connect_u:
                x = torch.cat([x, h.pop()], dim=1)
                x = self.skip_adapters[str(i)](x)
            if debug: self.save_x(x, i)
        return self.final(x)
        

class ConvBlock(nn.Module):

    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):

        assert(sampling in ["down", None, "up"])
        super(ConvBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling=="up" else 1
        stride = 2 if sampling=="down" else 1
        
        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()        
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.upscale(x)
        return self.net(x)

class ResidualBlock(nn.Module):

    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):

        assert(sampling in ["down", None, "up"])
        super(ResidualBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling=="up" else 1
        stride = 2 if sampling=="down" else 1

        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        if sampling == "down":
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=2, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        elif ch_in != ch_out:
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.fit = nn.Identity()
        
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upscale(x)
        out = self.fit(x) + self.net(x)       
        return self.act(out)

class SeparableConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):
        assert(sampling in["down", None, "up"])
        super(SeparableConvBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling == "up" else 1
        stride = 2 if sampling == "down" else 1
        
        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()        
        
        self.net = nn.Sequential(
            # Depthwise Convolution (conv each C separately), groups=ch_in is the key!
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, padding=padding, stride=stride, groups=ch_in, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.GELU(),
            
            # Pointwise Convolution (kernel 1x1, join Cs and make ch_out)
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.upscale(x)
        return self.net(x)


class SeparableResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):
        assert(sampling in ["down", None, "up"])
        super(SeparableResidualBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling == "up" else 1
        stride = 2 if sampling == "down" else 1

        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()

        self.net = nn.Sequential(
            # 1st block (stride/downsampling)
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, padding=padding, stride=stride, groups=ch_in, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.GELU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            
            # 2nd block (fixed w h)
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, stride=1, groups=ch_out, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        # local skip-connection
        if sampling == "down":
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=2, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        elif ch_in != ch_out:
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.fit = nn.Identity()
        
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upscale(x)
        out = self.fit(x) + self.net(x)       
        return self.act(out)        
        
class SEResidualBlock(nn.Module):

    def __init__(self, ch_in, ch_out, sampling=None, use_dropout=False, padding_mode='reflect'):

        assert(sampling in ["down", None, "up"])
        super(SEResidualBlock, self).__init__()
        
        kernel_size = 5 if sampling == "up" else 3
        padding = 2 if sampling=="up" else 1
        stride = 2 if sampling=="down" else 1

        self.upscale = nn.Upsample(size=None, scale_factor=(2,2)) if sampling == "up" else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        if sampling == "down":
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=2, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        elif ch_in != ch_out:
            self.fit = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, padding_mode=padding_mode, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.fit = nn.Identity()
        
        squeeze_ch = max(1, ch_out // 16) 
        self.squeeze_excitation = torchvision.ops.SqueezeExcitation(ch_out, squeeze_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upscale(x)
        out = self.fit(x) + self.squeeze_excitation(self.net(x))
        return self.act(out)
        

# ---------------------------------------------------------------------------------------------------------------- Attention modules


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self, B, H, W, device):
        mask = torch.zeros((H, H), device=device)
        mask.fill_diagonal_(-1e9) 
        # Expand to Batch*Width shape
        return mask.unsqueeze(0).expand(B * W, -1, -1)

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width, x.device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, in_dim, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.qkv = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=False)
        self.dropout = attn_dropout

    def forward(self, x, mask=None):
        B, C, H, W = x.size()

        qkv = self.qkv(x) # (B, 3*C, H, W)
        qkv = qkv.view(B, 3, C, H * W).permute(1, 0, 3, 2) # (3, B, H*W, C)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H*W, C)

        # H*W (pixels), Features (C)
        attn = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout)
        
        # restore image shape
        output = attn.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return output

class DualAttentionBlock(nn.Module):
    def __init__(self, ch):
        super(DualAttentionBlock, self).__init__()        
        self.pa = PositionAttention(ch)
        self.ca = ChannelAttentionDANet()

    def forward(self, x):    
        return self.pa(x) + self.ca(x)        

class PositionAttention(nn.Module):
    
    def __init__(self, in_dim, reduction_ratio=16):
        super(PositionAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)        

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class ChannelAttentionDANet(nn.Module):
    def __init__(self):
        super(ChannelAttentionDANet, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1) # (B, C, H*W)
        proj_key = x.view(B, C, -1).permute(0, 2, 1) # (B, H*W, C)
        
        # (B, C, C) - each C looks on every C
        energy = torch.bmm(proj_query, proj_key)
        # fix for NaN in Softmax
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = torch.softmax(energy_new, dim=-1)
        
        proj_value = x.view(B, C, -1)
        out = torch.bmm(attention, proj_value).view(B, C, H, W)
        return self.gamma * out + x
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # for both pools
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * out.sigmoid()
