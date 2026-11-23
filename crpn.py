import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _require_even_channels(channels: int) -> None:
    if channels < 2 or channels % 2:
        raise ValueError(f"Expected a positive even channel count, got {channels}.")


class PumpDD(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        _require_even_channels(channels)
        self.strided = nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1)
        self.reconstruct = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.shuffle = nn.ChannelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-2) % 2 or x.size(-1) % 2:
            raise ValueError("PumpDD requires even height and width.")
        local, detail = x.chunk(2, dim=1)
        patches = (
            detail[..., ::2, ::2],
            detail[..., ::2, 1::2],
            detail[..., 1::2, ::2],
            detail[..., 1::2, 1::2],
        )
        local = self.strided(local)
        detail = self.reconstruct(torch.cat(patches, dim=1))
        return self.shuffle(torch.cat((local, detail), dim=1))


class PumpU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest") * 0.5


class PumpDL(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        _require_even_channels(channels)
        half = channels // 2
        self.strided = nn.Conv2d(half, half, 3, stride=2, padding=1)
        self.enhanced = nn.Sequential(
            nn.Conv2d(half, half, 3, stride=2, padding=1),
            nn.Conv2d(half, half, 3, padding=1, groups=half),
            nn.GELU(),
        )
        self.shuffle = nn.ChannelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return self.shuffle(torch.cat((self.strided(x1), self.enhanced(x2)), dim=1))


def _fft_input(x: Tensor) -> Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class FrequencyCore(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.phi_f = nn.Conv2d(channels, channels, 1)
        self.phi_c = nn.Conv2d(channels, channels, 1)
        self.p = nn.Conv2d(channels, channels, 1)
        self.q = nn.Conv2d(channels, channels, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, u: Tensor) -> Tensor:
        weight = self.phi_f(self.gap(u))
        spectrum = torch.fft.fft2(_fft_input(u))
        amplitude = torch.fft.ifft2(spectrum * _fft_input(weight)).abs().to(u.dtype)
        f_a = amplitude * self.phi_c(self.gap(amplitude))

        p = _fft_input(self.p(u))
        q = _fft_input(self.q(u))
        modulated = p * torch.fft.fft2(q)
        f_b = torch.fft.ifft2(modulated).abs().to(u.dtype)
        return f_a + f_b


class SpatialCore(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.point = nn.Conv2d(channels, channels, 1, groups=channels)
        self.large = nn.Conv2d(channels, channels, 31, padding=15, groups=channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, u: Tensor) -> Tensor:
        return self.point(u) + self.large(u) + self.local(u)


class PumpCore(nn.Module):
    def __init__(self, channels: int, e: float = 0.25):
        super().__init__()
        if channels < 1 or not 0 < e <= 1 or not float(channels * e).is_integer():
            raise ValueError("Require channels > 0, 0 < e <= 1 and an integer e*channels.")
        enhanced = int(channels * e)
        kept = channels - enhanced
        self.preserved_channels = kept
        self.map_input = nn.Conv2d(channels, channels, 1)
        self.preserve = nn.Conv2d(kept, kept, 1) if kept else None
        self.input_conv = nn.Sequential(nn.Conv2d(enhanced, enhanced, 1), nn.GELU())
        self.frequency = FrequencyCore(enhanced)
        self.spatial = SpatialCore(enhanced)
        self.output_conv = nn.Conv2d(enhanced, enhanced, 1)

    def forward(self, x: Tensor) -> Tensor:
        mapped = self.map_input(x)
        cut = self.preserved_channels
        u = self.input_conv(mapped[:, cut:])
        mixed = self.spatial(u) + self.frequency(u)
        enhanced = self.output_conv(F.relu(mixed))
        if self.preserve is None:
            return enhanced
        return torch.cat((self.preserve(mapped[:, :cut]), enhanced), dim=1)
