# Cross-Domain Deepfake Detection — Generator-Invariant Detector

Frozen CLIP ViT-B/16 backbone + a class-conditional adversarial head (Gradient
Reversal Layer, applied only to fake-class samples) for cross-generator
generalizable deepfake detection. See `Cross-Domain Deepfake Detection -
Methodology (One-Pager).docx` / the Phase I slides for the full rationale —
this README is about actually running the code.

**Status:** all code below has been unit-tested (GRL gradient math, model
forward/backward shapes, dataset pipeline, and the full train→evaluate loop)
using a randomly-initialized tiny CLIP config and synthetic images, so you
know the mechanics are correct. It has **not** been run on real data or the
real pretrained CLIP weights, because that needs network access to
HuggingFace/Kaggle/GitHub and a GPU — do that part on Colab.

---

## 1. Setup (Colab)

```python
!pip install -q -r requirements.txt
```

Everything else (the pretrained CLIP weights) downloads automatically the
first time you run `train.py`, via `transformers`.

## 2. Get the data

You need three environments, each split into `real/` and `fake/`
subfolders of face images. **Use a small subset of each source below — a
few thousand images per class is plenty for a course-scope project and
keeps Colab download/training time sane.** The full public releases are
much bigger than you need (DiFF alone is 500,000+ images / ~140GB).

```
data/
  environment_a_stylegan/
    real/*.jpg      <- from Flickr-Faces-HQ (FFHQ)
    fake/*.jpg       <- StyleGAN-generated
  environment_b_diff/
    real/*.jpg       <- DiFF's pristine faces
    fake/*.jpg        <- DiFF's diffusion-generated faces (pick 1-2 of the 13 methods)
  environment_c_faceforensics/
    real/*.jpg        <- FaceForensics++ pristine frames
    fake/*.jpg          <- FaceForensics++ manipulated frames (face-swap/reenactment)
```

**Environment A — StyleGAN faces:**
Kaggle dataset ["140k Real and Fake Faces"](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
(by xhlulu). Already split into real/fake — just subsample a few thousand
of each into the folders above.

**Environment B — DiFF (diffusion faces):**
[github.com/xacheng1996/diff](https://github.com/xacheng1996/diff) (paper:
Cheng et al., "Diffusion Facial Forgery Detection," ACM MM 2024). Synthesized
images are on Google Drive (linked from the repo); pristine images require
filling out a short access form (also linked from the repo). Pick 1–2 of
the 13 generation methods rather than all of them — you don't need the
full benchmark for this project.

**Environment C — FaceForensics++ (the OOD test set):**
[github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics).
Access requires agreeing to their terms via the form linked in that repo.
Use the `Deepfakes` or `FaceSwap` manipulation subset plus the matching
`original` (pristine) frames — a few thousand extracted frames is enough.

If any of these access forms take a while to come through, a **temporary
stand-in** for Environment C that needs no approval: hold out a *different*
DiFF generation method than the ones you used for Environment B (still
diffusion, but a different specific method/condition) — not as strong a
generalization test as a genuinely different forgery mechanism, but lets
you get the whole pipeline running while you wait for FaceForensics++ access.

## 3. Train

```bash
# Row 3 of your ablation table (proposed): CLIP + LoRA + adversarial term
python src/train.py --data-root data --epochs 8 --adversarial 1

# Training with Self-Blended Images (SBI) pseudo-fake augmentation (recommended: --sbi-ratio 0.3):
python src/train.py --data-root data --epochs 8 --adversarial 1 --use-lora 1 --sbi-ratio 0.3

# Row 2 of your ablation table: frozen CLIP baseline, no adversarial term
python src/train.py --data-root data --epochs 8 --adversarial 0
```

### Self-Blended Images (`--sbi-ratio`)

The `--sbi-ratio` flag (default `0.0`, recommended starting value `0.3`) manufactures on-the-fly blending pseudo-forgeries per Shiohara & Yamasaki (CVPR 2022, *"Detecting Deepfakes with Self-Blended Images"*).

- **Why it exists:** Standard deepfake datasets contain full-synthesis fakes (GAN, diffusion). When evaluated on video face-swap benchmarks (FaceForensics++), detectors collapse (~50% chance level) because face swaps exhibit *local blending boundaries* rather than whole-image synthesis artifacts. SBI dynamically blends real face copies with perturbed color and geometry using soft landmark-feathered convex hulls.
- **Exclusion from adversarial loss:** SBI pseudo-fakes participate in the real/fake classification loss normally, but are explicitly masked out of the generator-ID adversarial loss (`adv_loss`) since they do not belong to any generator family.
- **OOD integrity:** SBI only samples real faces from **Environments A and B**. FaceForensics++ (**Environment C**) is strictly untouched and remains a pure, held-out out-of-distribution benchmark.


Each run writes a checkpoint (`checkpoints/model_adv.pt` or
`checkpoints/model_baseline.pt`) and a per-epoch training-history JSON.
LoRA is enabled by default: the pretrained CLIP weights stay frozen while
small LoRA adapters and the two classifier heads train. Set `--use-lora 0`
to train only the heads with a fully frozen backbone. Runtime depends on
the dataset, device, and selected configuration.

## 4. Evaluate — this produces the actual ablation table

```bash
python src/evaluate.py --data-root data \
    --checkpoint checkpoints/model_baseline.pt \
    --compare checkpoints/model_adv.pt \
    --jpeg-qualities 30 50 70
```

This prints (and saves as JSON) ID accuracy/F1/ROC-AUC, OOD accuracy/F1/
ROC-AUC, the generalization gap, and the JPEG-compression robustness sweep,
for both checkpoints side by side — i.e. rows 2 and 3 of the three-row
ablation table from the methodology doc. Add row 1 (your earlier RGB +
frequency-fusion model's numbers) by hand for the full comparison.

## 5. Demo

```bash
python src/app.py --checkpoint checkpoints/model_adv.pt --share
```

Launches a Gradio page (with `--share`, a public link — useful from Colab)
where you upload an image and get a Real/Fake prediction with confidence.

## 6. Plotting the ablation chart

`checkpoints/*.eval.json` has everything you need for the bar chart
described in the methodology doc (ID vs. OOD accuracy per configuration).
A quick starting point:

```python
import json, matplotlib.pyplot as plt

baseline = json.load(open("checkpoints/model_baseline.eval.json"))
adv = json.load(open("checkpoints/model_adv.eval.json"))

configs = ["Frozen CLIP\n(no adversarial)", "Frozen CLIP\n+ adversarial"]
id_acc = [baseline["id"]["accuracy"], adv["id"]["accuracy"]]
ood_acc = [baseline["ood"]["accuracy"], adv["ood"]["accuracy"]]

x = range(len(configs))
plt.bar([i - 0.15 for i in x], id_acc, width=0.3, label="ID accuracy")
plt.bar([i + 0.15 for i in x], ood_acc, width=0.3, label="OOD accuracy")
plt.xticks(list(x), configs)
plt.ylabel("Accuracy")
plt.legend()
plt.title("Generalization gap: baseline vs. class-conditional adversarial")
plt.savefig("checkpoints/ablation_chart.png", dpi=150, bbox_inches="tight")
```

---

## Project structure

```
src/
  app.py             Gradio image-classification demo
  dataset.py         Three-environment data loading and preprocessing
  evaluate.py        ID/OOD metrics, robustness checks, and ablation output
  evaluate_fusion.py Evaluation for the RGB/frequency fusion baseline
  fusion_model.py    RGB/frequency fusion baseline
  grl.py             Gradient Reversal Layer
  losses.py          Classification and generator-adversarial losses
  model.py           CLIP detector with optional LoRA adapters
  plot_ablation.py   Ablation chart generation
  sbi.py             Self-Blended Images augmentation
  train.py           CLIP detector training
  train_fusion.py    RGB/frequency baseline training
tests/
  test_pipeline.py   Offline mechanics and pipeline checks
checkpoints/           Evaluation histories and selected result figures
data/                  Local datasets (not tracked)
requirements.txt
```

## What's already verified vs. what you'll verify on Colab

| Verified here (no GPU / no internet needed) | Verify on Colab |
|---|---|
| GRL forward is identity, backward genuinely reverses+scales the gradient | Real CLIP weights load correctly |
| Backbone receives zero gradient when frozen; heads receive nonzero gradient | Real accuracy numbers on real data |
| Fake-only masking in the loss only ever touches fake-labeled rows | Training time per epoch on a T4 |
| Full train → checkpoint → evaluate → ablation-table pipeline runs without error | The actual generalization gap you're trying to measure |
| Dataset pipeline handles the 3-environment folder layout correctly | — |

Run `python -m pytest tests/` any time you change `grl.py`, `model.py`, or
`losses.py` to re-confirm the mechanics before spending Colab GPU time on it.
