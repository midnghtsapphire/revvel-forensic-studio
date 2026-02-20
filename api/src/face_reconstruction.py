"""
Revvel Forensic Studio — Face Reconstruction
Reconstruct faces from partial/degraded images using AI inpainting.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os


class FaceReconstructor:
    """Reconstruct faces from partial or degraded images."""

    def __init__(self):
        self.inpaint_radius = 3
        self.inpaint_method = cv2.INPAINT_TELEA

    def reconstruct_from_partial(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct face from partial image.

        Args:
            image: Input image (BGR)
            mask: Binary mask indicating missing regions (white = missing)

        Returns:
            Dictionary with reconstructed image and metadata
        """
        if mask is None:
            # Auto-detect missing regions (very dark or very bright areas)
            mask = self._auto_detect_missing_regions(image)

        # Inpaint missing regions
        reconstructed = cv2.inpaint(image, mask, self.inpaint_radius, self.inpaint_method)

        # Apply denoising
        reconstructed = cv2.fastNlMeansDenoisingColored(reconstructed, None, 10, 10, 7, 21)

        # Enhance details
        reconstructed = self._enhance_details(reconstructed)

        return {
            "success": True,
            "reconstructed": reconstructed,
            "mask_used": mask,
            "missing_percentage": (np.sum(mask > 0) / mask.size) * 100,
        }

    def reconstruct_from_degraded(self, image: np.ndarray) -> Dict[str, Any]:
        """Reconstruct face from degraded/low-quality image."""
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Sharpen
        sharpened = self._sharpen_image(denoised)

        # Enhance contrast
        enhanced = self._enhance_contrast(sharpened)

        # Super-resolution (simple upscaling with interpolation)
        height, width = enhanced.shape[:2]
        upscaled = cv2.resize(
            enhanced,
            (width * 2, height * 2),
            interpolation=cv2.INTER_CUBIC,
        )

        return {
            "success": True,
            "reconstructed": upscaled,
            "original_size": (width, height),
            "reconstructed_size": (width * 2, height * 2),
        }

    def reconstruct_from_masked(self, image: np.ndarray) -> Dict[str, Any]:
        """Reconstruct face from image with masks/obstructions."""
        # Detect face region
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return {"success": False, "reason": "No face detected"}

        # Focus on first detected face
        x, y, w, h = faces[0]
        face_region = image[y : y + h, x : x + w]

        # Detect mask/obstruction (dark regions in face area)
        mask = self._detect_mask_region(face_region)

        # Reconstruct masked region
        reconstructed_face = cv2.inpaint(
            face_region, mask, self.inpaint_radius, self.inpaint_method
        )

        # Replace face region in original image
        result_image = image.copy()
        result_image[y : y + h, x : x + w] = reconstructed_face

        return {
            "success": True,
            "reconstructed": result_image,
            "face_region": (x, y, w, h),
            "mask_detected": mask,
        }

    def enhance_facial_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhance facial features for better visibility."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Sharpen
        enhanced = self._sharpen_image(enhanced)

        return {
            "success": True,
            "enhanced": enhanced,
        }

    def _auto_detect_missing_regions(self, image: np.ndarray) -> np.ndarray:
        """Auto-detect missing or corrupted regions in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect very dark regions (likely missing)
        _, dark_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

        # Detect very bright regions (likely overexposed/corrupted)
        _, bright_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

        # Combine masks
        mask = cv2.bitwise_or(dark_mask, bright_mask)

        # Dilate to expand detected regions slightly
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _detect_mask_region(self, face_image: np.ndarray) -> np.ndarray:
        """Detect mask/obstruction in face region."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Detect dark regions (likely mask)
        _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Focus on lower half of face (where masks typically are)
        height = mask.shape[0]
        mask[: height // 2, :] = 0

        return mask

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image using unsharp masking."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _enhance_details(self, image: np.ndarray) -> np.ndarray:
        """Enhance fine details in image."""
        # Apply bilateral filter to preserve edges while smoothing
        filtered = cv2.bilateralFilter(image, 9, 75, 75)

        # Sharpen
        sharpened = self._sharpen_image(filtered)

        return sharpened

    def compare_faces(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Dict[str, Any]:
        """Compare two face images for similarity."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Resize to same size
        height = min(gray1.shape[0], gray2.shape[0])
        width = min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (width, height))
        gray2 = cv2.resize(gray2, (width, height))

        # Calculate structural similarity
        from skimage.metrics import structural_similarity as ssim

        similarity_score = ssim(gray1, gray2)

        # Calculate histogram correlation
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Combined score
        combined_score = (similarity_score + hist_correlation) / 2

        return {
            "similarity_score": float(similarity_score),
            "histogram_correlation": float(hist_correlation),
            "combined_score": float(combined_score),
            "match_confidence": "high" if combined_score > 0.8 else "medium" if combined_score > 0.6 else "low",
        }
