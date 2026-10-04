"""
Self-Blended Images (SBI) Pseudo-Fake Generator.

Based on:
    Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images", CVPR 2022.
    https://arxiv.org/abs/2204.08376

Why SBI is critical for cross-domain deepfake detection:
    Detectors trained only on entire-face synthesis (StyleGAN, diffusion) focus on
    generator-specific artifacts across the whole image canvas. They fail
    catastrophically (~50% chance level) on face-swap / reenactment benchmarks
    like FaceForensics++, where the forgery is a LOCAL BLENDING boundary between
    two face patches.

    SBI manufactures realistic pseudo-forgeries from REAL images alone by
    extracting landmark-bounded facial regions, perturbing source and target
    copies independently (geometric warping, color jitter, Gaussian blurring),
    and alpha-blending them using a feathered mask. The model learns to detect
    blending boundaries and local inconsistencies rather than generator signatures.
"""

from typing import Optional, Tuple
import logging
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

logger = logging.getLogger(__name__)

YUNET_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


class SBIGenerator:
    """
    Generates self-blended pseudo-fake images from a single real face image.
    Uses YuNet (OpenCV FaceDetectorYN) or MediaPipe to locate facial landmarks
    and produce soft convex-hull masks for alpha blending.
    """

    def __init__(self, blur_radius: float = 5.0, model_path: Optional[str] = None):
        self.blur_radius = blur_radius
        self.skipped_count = 0
        self.detector = None

        onnx_path = Path(model_path) if model_path else YUNET_DEFAULT_PATH
        if onnx_path.exists() and hasattr(cv2, 'FaceDetectorYN'):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=str(onnx_path),
                    config="",
                    input_size=(224, 224),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                logger.info("SBIGenerator initialized with YuNet FaceDetectorYN.")
            except Exception as e:
                logger.warning(f"Could not load FaceDetectorYN: {e}")

    def _get_landmarks(self, img_np: np.ndarray) -> Optional[np.ndarray]:
        """Returns landmark coordinates (N, 2) in pixel space, or None."""
        h, w = img_np.shape[:2]
        if self.detector is not None:
            self.detector.setInputSize((w, h))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            _, faces = self.detector.detect(img_bgr)
            if faces is not None and len(faces) > 0:
                face = faces[0]
                # 5 landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner
                landmarks_5 = face[4:14].reshape((5, 2))
                
                # Expand 5 landmarks into a tight facial hull around eyes, nose, mouth & jawline
                re, le, nose, rm, lm = landmarks_5
                eye_center = (re + le) / 2.0
                mouth_center = (rm + lm) / 2.0
                face_center = (eye_center + mouth_center) / 2.0

                # Form a rich 12-point facial boundary hull
                pts = [
                    re + np.array([-10, -15]),
                    eye_center + np.array([0, -20]),
                    le + np.array([10, -15]),
                    le + np.array([15, 0]),
                    lm + np.array([15, 5]),
                    mouth_center + np.array([0, 15]),
                    rm + np.array([-15, 5]),
                    re + np.array([-15, 0]),
                    nose,
                ]
                return np.array(pts, dtype=np.int32)

        # Fallback heuristic: centered face region if detector fails
        pad_w = int(w * 0.3)
        pad_h = int(h * 0.35)
        cx, cy = w // 2, h // 2
        contour = cv2.ellipse2Poly((cx, cy), (pad_w, pad_h), 0, 0, 360, 30)
        return contour

    def _perturb_color(self, img: Image.Image) -> Image.Image:
        """Applies independent brightness, contrast, and color temperature jitter."""
        if random.random() < 0.8:
            factor = random.uniform(0.75, 1.25)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if random.random() < 0.8:
            factor = random.uniform(0.75, 1.25)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if random.random() < 0.8:
            factor = random.uniform(0.75, 1.25)
            img = ImageEnhance.Color(img).enhance(factor)
        if random.random() < 0.4:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def _perturb_geometry(self, img: Image.Image) -> Image.Image:
        """Applies slight affine transformation (shear, translation, scale)."""
        w, h = img.size
        dx = random.uniform(-0.03, 0.03) * w
        dy = random.uniform(-0.03, 0.03) * h
        scale = random.uniform(0.97, 1.03)
        shear = random.uniform(-0.03, 0.03)

        matrix = (
            scale, shear, dx,
            shear, scale, dy
        )
        return img.transform((w, h), Image.Transform.AFFINE, matrix, resample=Image.Resampling.BILINEAR)

    def generate(self, img: Image.Image) -> Optional[Tuple[Image.Image, Image.Image]]:
        """
        Produces a self-blended pseudo-fake image and its blending mask.
        Returns:
            (sbi_image, mask_image) or None if no face is detected.
        """
        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)
        h, w = img_np.shape[:2]

        landmarks = self._get_landmarks(img_np)
        if landmarks is None or len(landmarks) < 3:
            self.skipped_count += 1
            return None

        # Build convex hull mask from landmarks
        hull = cv2.convexHull(landmarks)
        raw_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(raw_mask, hull, 255)

        # Soft feathered edges
        ksize = int(self.blur_radius * 4) | 1  # ensure odd integer
        soft_mask = cv2.GaussianBlur(raw_mask, (ksize, ksize), self.blur_radius)
        mask_pil = Image.fromarray(soft_mask, mode="L")

        # Create source and target perturbations
        source = self._perturb_color(self._perturb_geometry(img_rgb))
        target = img_rgb

        # Alpha composite source into target via the feathered mask
        sbi_img = Image.composite(source, target, mask_pil)
        return sbi_img, mask_pil


if __name__ == "__main__":
    sample_path = Path("data/environment_a_stylegan/real")
    images = list(sample_path.glob("*.jpg")) + list(sample_path.glob("*.png"))
    if not images:
        print("No sample images found in data/environment_a_stylegan/real.")
    else:
        img = Image.open(images[0]).convert("RGB")
        gen = SBIGenerator()
        res = gen.generate(img)
        if res is not None:
            sbi, mask = res
            combined = Image.new("RGB", (img.width * 3, img.height))
            combined.paste(img, (0, 0))
            combined.paste(mask.convert("RGB"), (img.width, 0))
            combined.paste(sbi, (img.width * 2, 0))
            debug_out = Path("sbi_sanity_check.png")
            combined.save(debug_out)
            print(f"Successfully saved {debug_out.resolve()} (Original | Mask | SBI Pseudo-Fake)")
        else:
            print("No face detected in sample image.")
