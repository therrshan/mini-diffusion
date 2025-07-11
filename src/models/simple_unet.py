import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 time_emb_dim: int = 128, base_channels: int = 64):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        self.down1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down4 = ResBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        
        self.middle = ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        
        self.up4 = ResBlock(base_channels * 16, base_channels * 4, time_emb_dim)
        self.up3 = ResBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.up2 = ResBlock(base_channels * 4, base_channels, time_emb_dim)
        self.up1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps)
        
        x1 = self.down1(x, time_emb)
        x2 = self.down2(self.maxpool(x1), time_emb)
        x3 = self.down3(self.maxpool(x2), time_emb)
        x4 = self.down4(self.maxpool(x3), time_emb)
        
        x = self.middle(self.maxpool(x4), time_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x4.shape[2:]:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = self.up4(torch.cat([x, x4], dim=1), time_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, x3], dim=1), time_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, x2], dim=1), time_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1), time_emb)
        
        return self.final_conv(x)