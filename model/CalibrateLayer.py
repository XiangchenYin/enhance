import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

import torch
import torch.nn as nn
import torch.nn.functional as F

class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25


        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel,stride,padding, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        # print(px.shape)
        # print(pk.shape)
        po = F.conv2d(px, pk,stride=stride,padding=padding, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel,stride,padding, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    # print(px.shape)
    # print(pk.shape)
    po = F.conv2d(px, pk, **kwargs, groups=batch,stride=stride,padding=padding)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    # print(po.shape)
    return po

class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, stride,padding, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        # print(kernel.shape)
        out = F.conv2d(x, kernel, **kwargs,stride=stride,padding=padding, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out

class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel,stride,padding, **kwargs):
        # print('------------------------------')
        if self.training:
            # print('------------------------------training')
            if self.use_slow:
                return xcorr_slow(x, kernel,stride,padding, kwargs)
            else:
                return xcorr_fast(x, kernel, stride,padding,kwargs)
        else:
            # print('-------------------------------no_train')
            return Corr.apply(x, kernel,stride,padding,1, kwargs)


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8,stride=1,padding=1, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num
        self.stride = stride
        self.padding = padding

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, stride=stride,padding=padding,**kwargs)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input):
        kernel = self.conv_kernel(input)
        # print(kernel.shape)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        # print(kernel.shape)
        output = self.corr(input, kernel,self.stride,self.padding, **self.kwargs)  # B x (r*out) x W x H
        # print(output.shape)
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        # print(output.shape)
        guide_feature = self.conv_guide(input)
        # print(guide_feature.shape)
        # print(output.shape)

        output = self.asign_index(output, guide_feature)
        # print(output.shape)

        return output







##########################################
# Global Learnable Attention for Single Image Super-Resolution(TPAMI23)

# Global Learnable Attention
# class GLA(nn.Module):

#     def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, conv=common.default_conv, res_scale=1):
#         super(GLA,self).__init__()
#         self.chunk_size = chunk_size
#         self.n_hashes = n_hashes
#         self.reduction = reduction
#         self.res_scale = res_scale
#         self.conv_match = common.BasicBlock(conv, channels, channels//reduction, k_size, bn=False, act=nn.ReLU(inplace=True))
#         self.conv_assembly = common.BasicBlock(conv, channels, channels, k_size, bn=False, act=nn.ReLU(inplace=True))
#         self.conv_assembly_fc = common.BasicBlock(conv, channels, channels, k_size, bn=False, act=nn.ReLU(inplace=True))
#         self.fc = nn.Sequential(
#             nn.Linear(channels, chunk_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(chunk_size, chunk_size)
#         )

#     # Super-Bit Locality-Sensitive Hashing
#     def SBLSH(self, hash_buckets, x):
#         #x: [N,H*W,C]
#         N = x.shape[0]
#         device = x.device

#         #generate random rotation matrix
#         rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets//2) #[1,C,n_hashes,hash_buckets//2]
#         # assert rotations_shape[1] > rotations_shape[2]*rotations_shape[3]
#         random_rotations = torch.nn.init.orthogonal_(torch.empty(x.shape[-1], hash_buckets))
#         for _ in range(self.n_hashes-1):
#             random_rotations = torch.cat([random_rotations, torch.nn.init.orthogonal_(torch.empty(x.shape[-1],hash_buckets))], dim=-1)
#         # Training under multi-gpu: random_rotations.cuda() -> random_rotations.to(x.device) (suggested by Breeze-Zero from github: https://github.com/laoyangui/DLSN/issues/2)
#         random_rotations = random_rotations.reshape(rotations_shape[0], rotations_shape[1], rotations_shape[2], hash_buckets).expand(N, -1, -1, -1).cuda() #[N, C, n_hashes, hash_buckets]
#         rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, H*W, hash_buckets]

#         #get hash codes
#         hash_codes = torch.argmax(rotated_vecs, dim=-1) #[N,n_hashes,H*W]

#         #add offsets to avoid hash codes overlapping between hash rounds
#         offsets = torch.arange(self.n_hashes, device=device)
#         offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
#         hash_codes = torch.reshape(hash_codes + offsets, (N, -1,)) #[N,n_hashes*H*W]

#         return hash_codes

#     def add_adjacent_buckets(self, x):
#         x_extra_back = torch.cat([x[:,:,-1:, ...], x[:,:,:-1, ...]], dim=2)
#         x_extra_forward = torch.cat([x[:,:,1:, ...], x[:,:,:1,...]], dim=2)
#         return torch.cat([x, x_extra_back,x_extra_forward], dim=3)

#     def forward(self, input):

#         N,_,H,W = input.shape
#         x_embed = self.conv_match(input).view(N,-1,H*W).contiguous().permute(0,2,1)
#         y_embed = self.conv_assembly(input).view(N,-1,H*W).contiguous().permute(0,2,1)
#         fc_embed = self.conv_assembly_fc(input).view(N,-1,H*W).contiguous().permute(0,2,1)
#         x_embed_extra_index = torch.arange(H * W).unsqueeze(0).unsqueeze(0).permute(0, 2, 1).cuda() # [1, HW, 1]

#         L,C = x_embed.shape[-2:]

#         #number of hash buckets/hash bits
#         hash_buckets = min(L//self.chunk_size + (L//self.chunk_size)%2, 128)

#         #get assigned hash codes/bucket number
#         hash_codes = self.SBLSH(hash_buckets, x_embed) #[N,n_hashes*H*W]
#         hash_codes = hash_codes.detach()

#         #group elements with same hash code by sorting
#         _, indices = hash_codes.sort(dim=-1) #[N,n_hashes*H*W]
#         _, undo_sort = indices.sort(dim=-1) #undo_sort to recover original order
#         mod_indices = (indices % L) #now range from (0->H*W)

#         x_embed_sorted = common.batched_index_select(x_embed, mod_indices) #[N,n_hashes*H*W,C]
#         y_embed_sorted = common.batched_index_select(y_embed, mod_indices) #[N,n_hashes*H*W,C]
#         fc_embed_embed_sorted = common.batched_index_select(fc_embed, mod_indices) #[N,n_hashes*H*W,C]

#         #pad the embedding if it cannot be divided by chunk_size
#         padding = self.chunk_size - L%self.chunk_size if L%self.chunk_size!=0 else 0
#         x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes,-1, C)) #[N, n_hashes, H*W,C]
#         y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))
#         fc_att_buckets = torch.reshape(fc_embed_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))

#         if padding:
#             pad_x = x_att_buckets[:,:,-padding:,:].clone()
#             pad_y = y_att_buckets[:,:,-padding:,:].clone()
#             pad_fc = fc_att_buckets[:,:,-padding:,:].clone()
#             x_att_buckets = torch.cat([x_att_buckets,pad_x],dim=2)
#             y_att_buckets = torch.cat([y_att_buckets,pad_y],dim=2)
#             fc_att_buckets = torch.cat([fc_att_buckets,pad_fc],dim=2)

#         x_att_buckets = torch.reshape(x_att_buckets,(N,self.n_hashes,-1,self.chunk_size,C)) #[N, n_hashes, num_chunks, chunk_size, C] # q
#         y_att_buckets = torch.reshape(y_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))
#         fc_att_buckets = torch.reshape(fc_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))

#         x_match = F.normalize(x_att_buckets, p=2, dim=-1,eps=5e-5)

#         #allow attend to adjacent buckets
#         x_match = self.add_adjacent_buckets(x_match) #[N, n_hashes, num_chunks, chunk_size*3, C]  # k
#         y_att_buckets = self.add_adjacent_buckets(y_att_buckets)
#         fc_att_buckets = self.add_adjacent_buckets(fc_att_buckets)
#         fc_raw_score = self.fc(fc_att_buckets).permute(0,1,2,4,3) #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

#         #unormalized attention score
#         raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match) + fc_raw_score #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

#         #softmax    self.sigmoid2(self.fc2(self.sigmoid1(self.fc1(x_att_buckets))))
#         bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
#         score = torch.exp(raw_score - bucket_score) #(after softmax)

#         ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets) #[N, n_hashes, num_chunks, chunk_size, C*self.reduction]
#         bucket_score = torch.reshape(bucket_score,[N,self.n_hashes,-1])
#         ret = torch.reshape(ret,(N,self.n_hashes,-1,C*self.reduction))

#         #if padded, then remove extra elements
#         if padding:
#             ret = ret[:,:,:-padding,:].clone()
#             bucket_score = bucket_score[:,:,:-padding].clone()

#         #recover the original order
#         ret = torch.reshape(ret, (N, -1, C*self.reduction)) #[N, n_hashes*H*W,C]
#         bucket_score = torch.reshape(bucket_score, (N, -1,)) #[N,n_hashes*H*W]
#         ret = common.batched_index_select(ret, undo_sort)#[N, n_hashes*H*W,C]
#         bucket_score = bucket_score.gather(1, undo_sort)#[N,n_hashes*H*W]

#         #weighted sum multi-round attention
#         ret = torch.reshape(ret, (N, self.n_hashes, L, C*self.reduction)) #[N, n_hashes*H*W,C]
#         bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
#         probs = nn.functional.softmax(bucket_score,dim=1)
#         ret = torch.sum(ret * probs, dim=1)

#         ret = ret.permute(0,2,1).view(N,-1,H,W).contiguous()*self.res_scale+input
#         return ret


# ## Global Learnable Attention-based Features Fusion Module (GLAFFM)
# class GLAFFM(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, chunk_size, n_hashes):
#         super(GLAFFM, self).__init__()
#         modules_body = []
#         modules_body = [
#             LFFB(conv, n_feat, kernel_size, 16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
#             for _ in range(n_resblocks)] # To stabilize the training, you can try lowering the value of res_scale.

#         modules_body.append(attention.GLA(channels=n_feat, chunk_size=chunk_size, n_hashes=n_hashes, reduction=reduction, res_scale=1)) # To stabilize the training, you can try lowering the value of res_scale.
#         modules_body.append(conv(n_feat, n_feat, kernel_size))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x
#         return res


## Local Features Fusion Block (LFFB)
class LFFB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(LFFB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res    

##########################################

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        
        
    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

    


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1, Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class CalibrateLayer(nn.Module):
    def __init__(self, dim):
        super(CalibrateLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(dim, dim * 2, 1)
        self.SFT_scale_conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.SFT_shift_conv0 = nn.Conv2d(dim, dim * 2, 1)
        self.SFT_shift_conv1 = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x), 0.1, inplace=True))
        return x * scale + shift

    
    
class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels*2),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels*2, out_channels=channels*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels*2),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta


    
class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels*2),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        illu = fea + input
        return illu


    
if __name__ == '__main__':
#     enhance = EnhanceNetwork(layers=1, channels=48)
    drconv2d = DRConv2d(48, 48, kernel_size=3, region_num=5, stride=1, padding=1)
    img = torch.ones(1, 48, 256, 256)
    print(drconv2d(img).shape)
    