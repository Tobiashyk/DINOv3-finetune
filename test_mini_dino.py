"""
Test script for Mini-DINOv3 implementation

This script tests the core components with dummy data to ensure everything works correctly.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.loss import DINOLoss
from src.model.ssl_ref import DINOHead, MiniDINO


def test_dino_loss():
    """Test DINOLoss with dummy data."""
    print("\n" + "="*50)
    print("Testing DINOLoss...")
    print("="*50)

    batch_size = 4
    out_dim = 1024

    # Create loss
    loss_fn = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

    # Create dummy data
    student_logits = torch.randn(batch_size, out_dim)
    teacher_logits = torch.randn(batch_size, out_dim)

    # Test Sinkhorn-Knopp
    teacher_probs = loss_fn.sinkhorn_knopp_teacher(teacher_logits, teacher_temp=0.04)

    # Check that probabilities sum to 1
    prob_sums = teacher_probs.sum(dim=1)
    print(f"✓ Teacher probability sums: {prob_sums.tolist()}")
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), "Probabilities should sum to 1"

    # Test loss computation
    loss = loss_fn(student_logits, teacher_probs)
    print(f"✓ Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"

    # Test center update
    loss_fn.update_center(teacher_logits)
    print(f"✓ Center shape: {loss_fn.center.shape}")
    assert loss_fn.center.shape == (1, out_dim), "Center should have correct shape"

    print("✓ DINOLoss tests passed!")


def test_dino_head():
    """Test DINOHead with dummy data."""
    print("\n" + "="*50)
    print("Testing DINOHead...")
    print("="*50)

    batch_size = 4
    in_dim = 384
    out_dim = 1024

    # Create head
    head = DINOHead(in_dim=in_dim, out_dim=out_dim)

    # Create dummy input
    x = torch.randn(batch_size, in_dim)

    # Forward pass
    output = head(x)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (batch_size, out_dim), "Output should have correct shape"

    # Check that gradients flow
    loss = output.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in head.parameters())
    print(f"✓ Gradients computed: {has_grad}")
    assert has_grad, "Gradients should be computed"

    print("✓ DINOHead tests passed!")


def test_mini_dino():
    """Test MiniDINO model with dummy data."""
    print("\n" + "="*50)
    print("Testing MiniDINO...")
    print("="*50)

    batch_size = 2
    image_size = 224
    out_dim = 1024

    # Create model (use smaller backbone for testing)
    print("Loading model (this may take a moment)...")
    model = MiniDINO(
        backbone_name='dinov3_vits14',
        out_dim=out_dim,
    )

    # Create dummy images
    images = torch.randn(batch_size, 3, image_size, image_size)

    # Forward pass
    print("Running forward pass...")
    outputs = model(images)

    print(f"✓ Loss: {outputs['loss'].item():.4f}")
    print(f"✓ Student logits shape: {outputs['student_logits'].shape}")
    print(f"✓ Teacher probs shape: {outputs['teacher_probs'].shape}")

    assert outputs['loss'].item() > 0, "Loss should be positive"
    assert outputs['student_logits'].shape == (batch_size, out_dim), "Student logits shape incorrect"
    assert outputs['teacher_probs'].shape == (batch_size, out_dim), "Teacher probs shape incorrect"

    # Test backward pass
    print("Testing backward pass...")
    outputs['loss'].backward()

    # Check student has gradients
    student_has_grad = any(p.grad is not None for p in model.student_backbone.parameters())
    print(f"✓ Student has gradients: {student_has_grad}")
    assert student_has_grad, "Student should have gradients"

    # Check teacher has no gradients
    teacher_has_grad = any(p.grad is not None for p in model.teacher_backbone.parameters())
    print(f"✓ Teacher has no gradients: {not teacher_has_grad}")
    assert not teacher_has_grad, "Teacher should not have gradients"

    # Test teacher update
    print("Testing teacher EMA update...")
    old_teacher_param = next(model.teacher_backbone.parameters()).clone()
    model.update_teacher(momentum=0.996)
    new_teacher_param = next(model.teacher_backbone.parameters())

    params_changed = not torch.equal(old_teacher_param, new_teacher_param)
    print(f"✓ Teacher parameters updated: {params_changed}")
    assert params_changed, "Teacher parameters should change after update"

    print("✓ MiniDINO tests passed!")


def test_training_step():
    """Test a complete training step."""
    print("\n" + "="*50)
    print("Testing complete training step...")
    print("="*50)

    batch_size = 2
    image_size = 224
    out_dim = 1024

    # Create model
    print("Loading model...")
    model = MiniDINO(backbone_name='dinov3_vits14', out_dim=out_dim)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.get_student_parameters(), lr=0.001)

    # Create dummy batch
    images = torch.randn(batch_size, 3, image_size, image_size)

    # Training step
    print("Running training step...")

    # Forward
    outputs = model(images)
    loss = outputs['loss']
    print(f"✓ Initial loss: {loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.get_student_parameters(), max_norm=3.0)
    print(f"✓ Gradient norm: {grad_norm:.4f}")

    # Optimizer step
    optimizer.step()

    # Update teacher
    model.update_teacher(momentum=0.996)

    # Run another forward pass to see if loss changes
    with torch.no_grad():
        outputs2 = model(images)
        loss2 = outputs2['loss']

    print(f"✓ Loss after step: {loss2.item():.4f}")
    print(f"✓ Loss changed: {abs(loss.item() - loss2.item()) > 1e-6}")

    print("✓ Training step test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Mini-DINOv3 Implementation Tests")
    print("="*60)

    try:
        # Test individual components
        test_dino_loss()
        test_dino_head()
        test_mini_dino()
        test_training_step()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour Mini-DINOv3 implementation is working correctly!")
        print("\nNext steps:")
        print("1. Prepare your training data")
        print("2. Run: python src/model/train_mini_dino.py --data_dir /path/to/images")
        print("3. Monitor the training loss")
        print("4. Use the trained model for feature extraction")

    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
