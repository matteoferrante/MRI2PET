import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

    
    
class DepthSepConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)


class DepthSepConvTranspose3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
    
    def forward(self,x):
        return self.depthwise_separable_conv(x)



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups= 4,depthwise_separable=False):
        
        super().__init__()
        self.norm1=nn.GroupNorm(n_groups,in_channels)
        self.act1=Swish()
        
        
        if depthwise_separable:
            self.conv1=DepthSepConv3d(in_channels,out_channels,kernel_size=(3,3,3),padding=(1,1,1))
            self.conv2 = DepthSepConv3d(out_channels, out_channels, kernel_size=(3, 3,3), padding=(1,1,1))

        else:
            self.conv1=nn.Conv3d(in_channels,out_channels,kernel_size=(3,3,3),padding=(1,1,1))
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3,3), padding=(1, 1,1))

        
        
        self.norm2=nn.GroupNorm(n_groups,out_channels)
        self.act2=Swish()
        
        if in_channels != out_channels:
            
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1))
#             if depthwise_separable:
#                 self.shortcut = DepthSepConv3d(in_channels, out_channels, kernel_size=(1,1,1))
#             else:
#                 self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1))
            
        else: 
            self.shortcut = nn.Identity()
        
        
    def forward(self,x):
        #x shape (bs,in_channels,h,w,d)
        
        h=self.conv1(self.act1(self.norm1(x)))
                
        h=self.conv2(self.act2(self.norm2(h)))
        
        
        return h+self.shortcut(x)
    
    
class AttentionBlock(nn.Module):
    def __init__(self,n_channels,n_heads=1,d_k=None,n_groups=4):
        super().__init__()
        
        if d_k is None:
            d_k=n_channels
            
        self.norm=nn.GroupNorm(n_groups,n_channels)
        self.q=nn.Linear(n_channels,n_heads*d_k) #for q,k,v
        self.k=nn.Linear(n_channels,n_heads*d_k) #for q,k,v
        self.v=nn.Linear(n_channels,n_heads*d_k) #for q,k,v
        
        self.attn=nn.MultiheadAttention(n_heads*d_k,n_heads)
        
        
        self.output=nn.Linear(n_heads*d_k,n_channels)
        
        self.scale=d_k**-0.5
        self.n_heads=n_heads
        self.d_k=d_k
        
    def forward(self,x):
        #x (bs,in_channel,h,w)
        
        
        batch_size, n_channels, height, width,depth = x.shape
        
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        q=self.q(x)
        k=self.k(x)
        v=self.v(x)
        
        attn_output, attn_output_weights=self.attn(q,k,v)
        res=self.output(attn_output)
        
        #skip connection
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width,depth)
        return res
    
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,has_attn,depthwise_separable=False):
        super().__init__()
        
        self.res=ResidualBlock(in_channels,out_channels,depthwise_separable=depthwise_separable)
        if has_attn:
            self.attn=AttentionBlock(out_channels)
        else:
            self.attn=nn.Identity()
            
    def forward(self,x):
        x=self.res(x)
        x=self.attn(x)
            
        return x
        
class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,has_attn,depthwise_separable=False):
        super().__init__()
        
        self.res=ResidualBlock(in_channels+out_channels,out_channels,depthwise_separable=depthwise_separable) #because of skip
        if has_attn:
            self.attn=AttentionBlock(out_channels)
        else:
            self.attn=nn.Identity()
        
        
    def forward(self,x):
        x=self.res(x)
        x=self.attn(x)

        return x
    
class MiddleBlock(nn.Module):
    def __init__(self,n_channels,depthwise_separable=False):
        super().__init__()
        #res attn res
        
        self.res1=ResidualBlock(n_channels,n_channels,depthwise_separable=depthwise_separable)
        self.attn=AttentionBlock(n_channels)
        self.res2=ResidualBlock(n_channels,n_channels,depthwise_separable=depthwise_separable)
        
    def forward(self,x):
        x=self.res1(x)
        x=self.attn(x)
        x=self.res2(x)
        
        return x
    
    
class Upsample(nn.Module):
    
    #upsample by factor two
    def __init__(self,n_channels,depthwise_separable=False):
        super().__init__()
        if depthwise_separable:
            self.conv=DepthSepConvTranspose3d(n_channels,n_channels,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1))
        else:
            self.conv=nn.ConvTranspose3d(n_channels,n_channels,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1))
        
    def forward(self,x):
        return self.conv(x)
    

class Downsample(nn.Module):
    
    #downsample by factor two
    def __init__(self,n_channels,depthwise_separable=False):
        super().__init__()
        
        if depthwise_separable:
            self.conv=DepthSepConv3d(n_channels,n_channels,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        else:
            self.conv=nn.Conv3d(n_channels,n_channels,kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        
    def forward(self,x):
        return self.conv(x)
    
    
    
class UNet3D(nn.Module):
    def __init__(self,image_channels=3,n_channels=64,ch_mults=(1,2,2,4),is_attn=
                (False,False,True,True),n_blocks=2,depthwise_separable=False,sigmoid=True):
        
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        n_resolutions=len(ch_mults) #number of resolutions
        
        #project 
#         if depthwise_separable:
#             self.image_proj=DepthSepConv3d(image_channels,n_channels,kernel_size=(3,3,3),padding=(1,1,1))
        
        # else:
        #     self.image_proj=nn.Conv3d(image_channels,n_channels,kernel_size=(3,3,3),padding=(1,1,1))
        self.image_proj=nn.Conv3d(image_channels,n_channels,kernel_size=(3,3,3),padding=(1,1,1))
        
        #first half
        down=[]
        out_channels=in_channels=n_channels
        
        #for each resolution
        for i in range(n_resolutions):
            out_channels=in_channels*ch_mults[i]
            
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels,out_channels,is_attn[i],depthwise_separable))
                in_channels=out_channels
            
            #downsample (not the last)
            if i<n_resolutions-1:
                down.append(Downsample(in_channels,depthwise_separable=False))
                
        self.down=nn.ModuleList(down)
        
        self.middle=MiddleBlock(out_channels)
        
        #up
        
        up=[]
        in_channels=out_channels
        
        for i in reversed(range(n_resolutions)):
            out_channels=in_channels
            
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels,out_channels,is_attn[i],depthwise_separable=depthwise_separable))
            
            out_channels=in_channels//ch_mults[i] #reduce number of channels while increasing image size
            #final block
            up.append(UpBlock(in_channels,out_channels,is_attn[i],depthwise_separable=depthwise_separable))
            in_channels=out_channels
            
            if i>0:
                up.append(Upsample(in_channels,depthwise_separable=False))
                
        self.up=nn.ModuleList(up)
        
        self.norm=nn.GroupNorm(8,n_channels)
        self.act=Swish()
        if depthwise_separable:
            self.final=DepthSepConv3d(in_channels,image_channels,kernel_size=(3,3,3),padding=(1,1,1))
        
        else:
            self.final=nn.Conv3d(in_channels,image_channels,kernel_size=(3,3,3),padding=(1,1,1))
        
    def forward(self,x):
        
        #project image
        x=self.image_proj(x)
        
        h=[x] #store output for skip connections
        
        for m in self.down:
            x=m(x)
            h.append(x)
                
        x=self.middle(x)
        
        for m in self.up:
            if isinstance(m,Upsample):
                x=m(x)
            else:
                #skip connection
                
                
                s=h.pop()
                
                
                x=torch.cat((x,s),dim=1)
                
                x=m(x)
                
        out=self.final(self.act(self.norm(x)))
        if self.sigmoid:
            out=self.sigmoid(out)
        return out
        
        