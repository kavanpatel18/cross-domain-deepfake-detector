"""
Offline sanity tests — no internet, no GPU, no real CLIP weights required.

These confirm the MECHANICS are correct (gradient reversal math, tensor
shapes, masking logic, the full train/eval pipeline running without
crashing). They do NOT tell you anything about real-world accuracy —
that only comes from running on Colab with real weights and real data.

Run with:  python -m pytest tests/ -v
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True, scope="module")
def patch_clip():
    """Swap in a tiny randomly-initialized CLIP so tests run without network access."""
    from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

    tiny_cfg = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, image_size=224, patch_size=16,
    )
    orig_vision = CLIPVisionModel.from_pretrained
    orig_proc = CLIPImageProcessor.from_pretrained

    CLIPVisionModel.from_pretrained = staticmethod(lambda name: CLIPVisionModel(tiny_cfg))
    CLIPImageProcessor.from_pretrained = staticmethod(lambda name: CLIPImageProcessor(
        do_resize=True, size={"shortest_edge": 224},
        do_center_crop=True, crop_size={"height": 224, "width": 224},
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    ))
    yield
    CLIPVisionModel.from_pretrained = orig_vision
    CLIPImageProcessor.from_pretrained = orig_proc


@pytest.fixture(scope="module")
def synthetic_data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for env in ["environment_a_stylegan", "environment_b_diff", "environment_c_faceforensics"]:
        for cls in ["real", "fake"]:
            d = root / env / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(6):
                img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
                img.save(d / f"{i}.jpg", quality=90)
    yield root
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- GRL ----

class TestGRL:
    def test_forward_is_identity(self):
        from grl import GradientReversalLayer
        x = torch.randn(4, 8, requires_grad=True)
        y = GradientReversalLayer(lambda_=1.0)(x)
        assert torch.allclose(x, y)

    def test_backward_reverses_and_scales_gradient(self):
        from grl import GradientReversalLayer
        x = torch.randn(4, 8, requires_grad=True)
        y = GradientReversalLayer(lambda_=0.7)(x)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, -0.7))

    def test_lambda_schedule_ramps_zero_to_near_one(self):
        from grl import grl_lambda_schedule
        assert grl_lambda_schedule(0.0) == 0.0
        assert grl_lambda_schedule(1.0) > 0.9
        # monotonically increasing
        vals = [grl_lambda_schedule(p) for p in [0, 0.25, 0.5, 0.75, 1.0]]
        assert vals == sorted(vals)


# ------------------------------------------------------------- Model -----

class TestModel:
    def test_forward_shapes(self):
        from model import GeneratorInvariantDetector, ModelConfig
        cfg = ModelConfig(num_generators=2, head_hidden_dim=16)
        net = GeneratorInvariantDetector(cfg)
        pixel_values = torch.randn(5, 3, 224, 224)
        rf_logits, gen_logits = net(pixel_values)
        assert rf_logits.shape == (5, 2)
        assert gen_logits.shape == (5, 2)

    def test_backbone_frozen_heads_trainable(self):
        from model import GeneratorInvariantDetector, ModelConfig
        from losses import compute_loss
        cfg = ModelConfig(num_generators=2, head_hidden_dim=16, freeze_backbone=True)
        net = GeneratorInvariantDetector(cfg)

        pixel_values = torch.randn(6, 3, 224, 224)
        rf_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        gen_labels = torch.tensor([0, 0, 0, 0, 1, 1])

        rf_logits, gen_logits = net(pixel_values)
        out = compute_loss(rf_logits, gen_logits, rf_labels, gen_labels, lambda_t=0.5)
        net.zero_grad()
        out.total.backward()

        assert all(p.grad is None for p in net.backbone.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in net.real_fake_head.parameters())


# ------------------------------------------------------------ Losses -----

class TestLosses:
    def test_adversarial_term_only_uses_fake_rows(self):
        from losses import compute_loss
        rf_logits = torch.randn(4, 2)
        gen_logits = torch.randn(4, 2)
        rf_labels = torch.tensor([0, 0, 1, 1])   # 2 real, 2 fake
        gen_labels = torch.tensor([0, 0, 0, 1])

        out = compute_loss(rf_logits, gen_logits, rf_labels, gen_labels, lambda_t=1.0)
        assert out.n_fake_in_batch == 2

    def test_no_fake_in_batch_does_not_crash(self):
        from losses import compute_loss
        rf_logits = torch.randn(3, 2)
        gen_logits = torch.randn(3, 2)
        rf_labels = torch.tensor([0, 0, 0])   # all real
        gen_labels = torch.tensor([0, 0, 0])

        out = compute_loss(rf_logits, gen_logits, rf_labels, gen_labels, lambda_t=1.0)
        assert out.n_fake_in_batch == 0
        assert torch.isfinite(out.total)

    def test_sbi_masked_out_of_adversarial_loss(self):
        from losses import compute_loss
        rf_logits = torch.randn(4, 2)
        gen_logits = torch.randn(4, 2)
        rf_labels = torch.tensor([0, 1, 1, 1])   # 1 real, 3 fake
        gen_labels = torch.tensor([0, 0, 1, 1])
        is_sbi = torch.tensor([False, False, True, True])  # 2 SBI fakes, 1 standard fake

        out = compute_loss(rf_logits, gen_logits, rf_labels, gen_labels, lambda_t=1.0, is_sbi=is_sbi)
        # Even though there are 3 fakes, the 2 SBI fakes are masked out
        assert out.n_fake_in_batch == 1



# ----------------------------------------------------------- Dataset -----

class TestDataset:
    def test_three_environment_split_sizes(self, synthetic_data_root):
        from dataset import build_train_val_test
        train_ds, val_ds, test_ds = build_train_val_test(synthetic_data_root, val_fraction=0.2, seed=0)
        # env A + env B = 2 envs * 2 classes * 6 images = 24 total for train+val
        assert len(train_ds) + len(val_ds) == 24
        # env C alone = 2 classes * 6 images = 12
        assert len(test_ds) == 12

    def test_batch_shapes(self, synthetic_data_root):
        from dataset import build_train_val_test
        from torch.utils.data import DataLoader

        train_ds, _, _ = build_train_val_test(synthetic_data_root, val_fraction=0.2, seed=0)
        batch = next(iter(DataLoader(train_ds, batch_size=4)))
        assert batch["pixel_values"].shape[1:] == (3, 224, 224)
        assert batch["real_fake_label"].dtype == torch.long
        assert "is_sbi" in batch
        assert batch["is_sbi"].dtype == torch.bool

    def test_sbi_dataset_expansion_and_isolation(self, synthetic_data_root):
        from dataset import build_train_val_test
        train_ds, val_ds, test_ds = build_train_val_test(
            synthetic_data_root, val_fraction=0.2, sbi_ratio=0.5, seed=0
        )
        # Baseline pool without SBI is 24 images
        # With sbi_ratio=0.5, 12 SBI samples added -> total train_pool=36
        assert len(train_ds) + len(val_ds) == 36
        sbi_count = sum(1 for s in train_ds.samples if s.is_sbi) + sum(1 for s in val_ds.samples if s.is_sbi)
        assert sbi_count == 12

        # Crucial check: Environment C (test split) must NEVER contain any SBI samples
        assert all(not s.is_sbi for s in test_ds.samples)
        assert len(test_ds) == 12


# ------------------------------------------------------- Full pipeline ---

class TestFullPipeline:
    def test_train_then_evaluate_end_to_end(self, synthetic_data_root, tmp_path):
        import train as train_mod
        import evaluate as eval_mod

        out_dir = tmp_path / "checkpoints"

        sys.argv = ["train.py", "--data-root", str(synthetic_data_root),
                    "--epochs", "1", "--batch-size", "4", "--adversarial", "1",
                    "--out-dir", str(out_dir)]
        train_mod.main()

        ckpt = out_dir / "model_adv.pt"
        assert ckpt.exists()

        sys.argv = ["evaluate.py", "--data-root", str(synthetic_data_root),
                    "--checkpoint", str(ckpt), "--jpeg-qualities", "50"]
        eval_mod.main()

        eval_json = ckpt.with_suffix(".eval.json")
        assert eval_json.exists()


# ------------------------------------------------- Fusion baseline -------

class TestFusionModel:
    def test_rgb_to_log_fft_shape(self):
        from fusion_model import rgb_to_log_fft
        x = torch.rand(4, 3, 224, 224)
        out = rgb_to_log_fft(x)
        assert out.shape == (4, 3, 224, 224), "FFT output must match input spatial dims"
        assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-5, "FFT output must be in [0,1]"

    def test_fusion_forward_shapes(self):
        from fusion_model import FusionBaseline
        model = FusionBaseline(pretrained=False)
        x = torch.rand(5, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (5, 2), "FusionBaseline must output (B, 2) logits"

    def test_fusion_backward_no_crash(self):
        from fusion_model import FusionBaseline
        model = FusionBaseline(pretrained=False)
        x = torch.rand(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()  # must not raise
        assert torch.isfinite(loss)


# ---------------------------------------------------- SBI Generator -------

class TestSBIGenerator:
    def test_sbi_generator_output_shape_and_range(self):
        from sbi import SBIGenerator
        gen = SBIGenerator()
        # Synthetic RGB image
        img = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype(np.uint8))
        res = gen.generate(img)

        # Either returns (sbi_image, mask_image) of exact size or None gracefully
        if res is not None:
            sbi, mask = res
            assert isinstance(sbi, Image.Image)
            assert isinstance(mask, Image.Image)
            assert sbi.size == (128, 128)
            assert mask.size == (128, 128)
            assert mask.mode == "L"

    def test_sbi_generator_handles_blank_or_invalid_gracefully(self):
        from sbi import SBIGenerator
        gen = SBIGenerator()
        # Flat black image has no face features
        black_img = Image.new("RGB", (64, 64), color="black")
        res = gen.generate(black_img)
        # Must return None without raising an exception or crashing
        # or return a valid synthetic blend
        if res is None:
            assert gen.skipped_count >= 1
        else:
            sbi, mask = res
            assert sbi.size == (64, 64)


