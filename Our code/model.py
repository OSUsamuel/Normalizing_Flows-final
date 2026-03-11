import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    """Positional encdoing for scalar time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-torch.arange(half).float() * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        args = t[:, None] * self.freqs[None]          # (B, half)
        emb = torch.cat([args.sin(), args.cos()], -1)  # (B, dim)
        return self.proj(emb)                          # (B, dim)


class ResBlock(nn.Module):
    """Residual block conditioned on time embedding."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, time_dim)
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.down(x), x          # return (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResBlock(out_ch + skip_ch, time_dim)
        self.proj = nn.Conv2d(out_ch + skip_ch, out_ch, 1)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return self.proj(x)

""" U-net velocity network """
class VelocityUNet(nn.Module): 
  """ should predict velocity v(x_t, t) for Rectified Flow on 28x28 grayscale
  images. 
  Archeticture: a shallow U-net with three resolution levels. 
  Channels: 32 -> 64 -> 128, time confitioning occurs at every residual block
  """
  def __init__(self, in_channels: int = 1, base_ch: int = 32, time_dim: int = 128):
    super().__init__()
    self.time_emb = SinusoidalTimeEmbedding(time_dim)

    #actual encoder
    self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
    # 28 -> 14
    self.down1 = DownBlock(base_ch,      base_ch * 2, time_dim)
    # 14 -> 7
    self.down2 = DownBlock(base_ch * 2,  base_ch * 4, time_dim)

    #bottleneck 
    self.mid1 = ResBlock(base_ch * 4, time_dim)
    self.mid2 = ResBlock(base_ch * 4, time_dim)

    # decoder
    # 7 -> 14
    self.up1 = UpBlock(base_ch * 4, base_ch * 2, time_dim)
    # 14 -> 28
    self.up2 = UpBlock(base_ch * 2, base_ch,     base_ch,     time_dim)
    






