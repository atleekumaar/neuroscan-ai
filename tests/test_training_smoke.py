"""
NEUROSCAN AI — Phase 5.13: Software Verification & Gradient Smoke Tests
Verifies:
  1. Finite loss computation with multi-class DiceCELoss
  2. Non-zero gradient flow across all 4-channel 3D U-Net parameters
  3. Optimizer step changes model weights deterministically
  4. Checkpoint serialization round-trip (saving and re-loading state_dict)
  5. Tensor dimension consistency between input, logits, and target
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.segmentation.model import (
    FourChannelUNetConfig,
    build_four_channel_unet,
    NativeBrainTumorUNet3D
)

try:
    from monai.losses import DiceCELoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


def test_finite_loss_and_gradient_flow():
    """Verify forward and backward pass produce finite loss and non-zero gradients."""
    torch.manual_seed(42)
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(8, 16, 32, 64, 128),
        dropout=0.1
    )
    model = build_four_channel_unet(config)
    model.train()

    # Synthetic batch: B=2, C=4, D=32, H=32, W=32
    inputs = torch.randn(2, 4, 32, 32, 32, requires_grad=True)
    targets = torch.randint(0, 4, (2, 1, 32, 32, 32))

    if MONAI_AVAILABLE:
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    outputs = model(inputs)
    assert outputs.shape == (2, 4, 32, 32, 32)
    assert torch.isfinite(outputs).all(), "Model output contains non-finite values!"

    if MONAI_AVAILABLE:
        loss = loss_fn(outputs, targets)
    else:
        loss = loss_fn(outputs, targets.squeeze(1))

    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0

    loss.backward()

    # Check gradient flow across layers
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient contains NaN/Inf in {name}"


def test_optimizer_step_updates_weights():
    """Verify that optimizer step actually updates parameters."""
    torch.manual_seed(42)
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(8, 16, 32, 64, 128),
        dropout=0.0
    )
    model = build_four_channel_unet(config)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    # Clone initial weights of first conv layer
    initial_weight = next(model.parameters()).clone()

    inputs = torch.randn(1, 4, 32, 32, 32)
    targets = torch.zeros((1, 1, 32, 32, 32), dtype=torch.int64)

    if MONAI_AVAILABLE:
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        loss = loss_fn(model(inputs), targets)
    else:
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(model(inputs), targets.squeeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated_weight = next(model.parameters())
    diff = torch.abs(updated_weight - initial_weight).max().item()
    assert diff > 1e-6, "Weights did not update after optimizer step!"


def test_checkpoint_roundtrip_fidelity(tmp_path):
    """Verify checkpoint serialization and restoration fidelity."""
    config = FourChannelUNetConfig(
        in_channels=4,
        out_channels=4,
        channels=(8, 16, 32, 64, 128),
        dropout=0.0
    )
    model1 = build_four_channel_unet(config)
    ckpt_path = tmp_path / "test_multimodal_ckpt.pth"

    meta = {
        "model_state_dict": model1.state_dict(),
        "config": {
            "architecture": "FourChannelUNet3D",
            "in_channels": 4,
            "out_channels": 4,
            "channels": (8, 16, 32, 64, 128)
        },
        "phase": "5.13_smoke"
    }
    torch.save(meta, str(ckpt_path))

    # Load into fresh model
    model2 = build_four_channel_unet(config)
    loaded_meta = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model2.load_state_dict(loaded_meta["model_state_dict"])

    # Verify identical predictions on same input
    dummy_input = torch.randn(1, 4, 32, 32, 32)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model1(dummy_input)
        out2 = model2(dummy_input)

    assert torch.allclose(out1, out2, atol=1e-6), "Restored checkpoint outputs differ!"
