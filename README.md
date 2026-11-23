# CRPN

This repository contains part of the core code for *CRPN: A Spatial-Frequency Core Enhanced Feature Fusion Network for Multiscale Dense Small Object Detection*.

CRPN uses P3 as the central scale for feature fusion. It brings spatial detail from P2 and semantic information from P4 and P5 into P3, enhances the fused features in the spatial and frequency domains, and propagates the enhanced features to P4 and P5. Detection is performed at P3, P4, and P5.

## Core Modules

The following PyTorch modules are available in `crpn.py` and can be used independently.

| Module | Function |
| --- | --- |
| `PumpDD` | Transfers spatial detail from P2 to P3 through strided convolution and spatial rearrangement. |
| `PumpU` | Upsamples features by a factor of two and scales their amplitudes by one half to preserve the L2 norm. |
| `PumpCore` | Enhances a subset of P3 channels through spatial and frequency-domain processing. The default enhancement ratio is `e=0.25`. |
| `PumpDL` | Propagates enhanced features from P3 to P4 and from P4 to P5 through two downsampling branches. |

## Usage

The modules require PyTorch and accept tensors in `[batch, channels, height, width]` format.

```python
import torch
from crpn_modules import PumpCore

x = torch.randn(1, 64, 80, 80)
y = PumpCore(channels=64, e=0.25)(x)
```
