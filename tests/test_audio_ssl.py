"""Torch-free tests for A2 SSL masking (the audio-MAE patch masker)."""
import numpy as np

from poultryai.tasks.audio.ssl import random_patch_mask, batch_patch_mask


def test_mask_shape_and_dtype():
    m = random_patch_mask(64, 250, rng=np.random.default_rng(0))
    assert m.shape == (64, 250) and m.dtype == bool


def test_mask_ratio_approximately_matched():
    m = random_patch_mask(64, 256, patch_mel=8, patch_frame=8, mask_ratio=0.6,
                          rng=np.random.default_rng(0))
    # patch grid divides evenly here → ratio should be close to target
    assert 0.5 < m.mean() < 0.7


def test_mask_is_deterministic_with_seed():
    a = random_patch_mask(64, 250, rng=np.random.default_rng(7))
    b = random_patch_mask(64, 250, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_zero_ratio_masks_nothing_full_ratio_masks_all():
    none = random_patch_mask(32, 64, mask_ratio=0.0, rng=np.random.default_rng(1))
    allm = random_patch_mask(32, 64, mask_ratio=1.0, rng=np.random.default_rng(1))
    assert none.sum() == 0 and allm.all()


def test_patches_are_contiguous_blocks():
    # a single masked patch should be an 8x8 contiguous block, not scattered pixels
    m = random_patch_mask(16, 16, patch_mel=8, patch_frame=8, mask_ratio=0.25,
                          rng=np.random.default_rng(3))
    # exactly one of the four 8x8 quadrants is fully masked
    quads = [m[:8, :8], m[:8, 8:], m[8:, :8], m[8:, 8:]]
    full = [q.all() for q in quads]
    assert sum(full) == 1 and m.sum() == 64


def test_batch_mask_shape_and_independence():
    b = batch_patch_mask(4, 64, 128, mask_ratio=0.5, rng=np.random.default_rng(2))
    assert b.shape == (4, 1, 64, 128)
    assert set(np.unique(b).tolist()) <= {0.0, 1.0}
    # masks differ across items (independent draws)
    assert not np.array_equal(b[0], b[1])
