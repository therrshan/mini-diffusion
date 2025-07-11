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

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(77, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=2
        )
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        seq_len = text_tokens.shape[1]
        pos_ids = torch.arange(seq_len, device=text_tokens.device).unsqueeze(0).expand(text_tokens.shape[0], -1)
        
        text_emb = self.embedding(text_tokens)
        pos_emb = self.position_embedding(pos_ids)
        
        x = text_emb + pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.final_proj(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.heads
        
        q = self.to_q(x).reshape(b, n, h, c // h).transpose(1, 2)
        k = self.to_k(context).reshape(b, -1, h, c // h).transpose(1, 2)
        v = self.to_v(context).reshape(b, -1, h, c // h).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.to_out(out)

class ConditionedResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, text_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.text_mlp = nn.Linear(text_emb_dim, out_channels)
        
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
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        time_emb = self.time_mlp(time_emb)
        text_emb = self.text_mlp(text_emb)
        
        h = h + time_emb[:, :, None, None] + text_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class ConditionedUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 time_emb_dim: int = 128, text_emb_dim: int = 128, 
                 base_channels: int = 64, vocab_size: int = None, tokenizer=None):
        super().__init__()
        
        if vocab_size is None and tokenizer is not None:
            vocab_size = tokenizer.vocab_size
        elif vocab_size is None:
            vocab_size = 30522  # Default to BERT vocab size
            
        self.vocab_size = vocab_size
        
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.text_encoder = TextEncoder(vocab_size, text_emb_dim)
        
        self.down1 = ConditionedResBlock(in_channels, base_channels, time_emb_dim, text_emb_dim)
        self.down2 = ConditionedResBlock(base_channels, base_channels * 2, time_emb_dim, text_emb_dim)
        self.down3 = ConditionedResBlock(base_channels * 2, base_channels * 4, time_emb_dim, text_emb_dim)
        self.down4 = ConditionedResBlock(base_channels * 4, base_channels * 8, time_emb_dim, text_emb_dim)
        
        self.middle = ConditionedResBlock(base_channels * 8, base_channels * 8, time_emb_dim, text_emb_dim)
        
        self.up4 = ConditionedResBlock(base_channels * 16, base_channels * 4, time_emb_dim, text_emb_dim)
        self.up3 = ConditionedResBlock(base_channels * 8, base_channels * 2, time_emb_dim, text_emb_dim)
        self.up2 = ConditionedResBlock(base_channels * 4, base_channels, time_emb_dim, text_emb_dim)
        self.up1 = ConditionedResBlock(base_channels * 2, base_channels, time_emb_dim, text_emb_dim)
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps)
        text_emb = self.text_encoder(text_tokens)
        
        x1 = self.down1(x, time_emb, text_emb)
        x2 = self.down2(self.maxpool(x1), time_emb, text_emb)
        x3 = self.down3(self.maxpool(x2), time_emb, text_emb)
        x4 = self.down4(self.maxpool(x3), time_emb, text_emb)
        
        x = self.middle(self.maxpool(x4), time_emb, text_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x4.shape[2:]:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = self.up4(torch.cat([x, x4], dim=1), time_emb, text_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, x3], dim=1), time_emb, text_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, x2], dim=1), time_emb, text_emb)
        
        x = self.upsample(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1), time_emb, text_emb)
        
        return self.final_conv(x)