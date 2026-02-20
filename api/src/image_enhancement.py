"""
Revvel Forensic Studio — Image Enhancement
Deblur, denoise, super-resolution, and forensic-grade image enhancement.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


class ImageEnhancer:
    """Advanced image enhancement for forensic analysis."""

    def deblur(
        self, image: np.ndarray, kernel_size: int = 5, strength: float = 1.0
    ) -> Dict[str, Any]:
        """
        Deblur image using Wiener deconvolution approximation.

        Args:
            image: Input image (BGR)
            kernel_size: Size of deblurring kernel
            strength: Deblurring strength (0.5-2.0)

        Returns:
            Dictionary with deblurred image
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        # Create motion blur kernel (simulating camera shake)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Apply deconvolution using filter2D with inverse kernel
        deblurred = cv2.filter2D(img_float, -1, kernel * strength)

        # Clip and convert back
        deblurred = np.clip(deblurred, 0, 1)
        deblurred = (deblurred * 255).astype(np.uint8)

        # Apply sharpening
        deblurred = self._sharpen(deblurred, strength=1.5)

        return {
            "success": True,
            "deblurred": deblurred,
            "kernel_size": kernel_size,
            "strength": strength,
        }

    def denoise(
        self, image: np.ndarray, method: str = "nlm", strength: int = 10
    ) -> Dict[str, Any]:
        """
        Denoise image using various methods.

        Args:
            image: Input image (BGR)
            method: Denoising method ("nlm", "bilateral", "gaussian")
            strength: Denoising strength

        Returns:
            Dictionary with denoised image
        """
        if method == "nlm":
            # Non-Local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, strength, strength, 7, 21
            )
        elif method == "bilateral":
            # Bilateral Filter (preserves edges)
            denoised = cv2.bilateralFilter(image, 9, strength * 2, strength * 2)
        elif method == "gaussian":
            # Gaussian Blur
            denoised = cv2.GaussianBlur(image, (5, 5), strength / 10)
        else:
            return {"success": False, "error": f"Unknown method: {method}"}

        return {
            "success": True,
            "denoised": denoised,
            "method": method,
            "strength": strength,
        }

    def super_resolution(
        self, image: np.ndarray, scale: int = 2
    ) -> Dict[str, Any]:
        """
        Upscale image using super-resolution techniques.

        Args:
            image: Input image (BGR)
            scale: Upscaling factor (2, 3, or 4)

        Returns:
            Dictionary with upscaled image
        """
        height, width = image.shape[:2]

        # Use INTER_CUBIC for better quality upscaling
        upscaled = cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        )

        # Apply edge-preserving filter
        upscaled = cv2.edgePreservingFilter(upscaled, flags=1, sigma_s=60, sigma_r=0.4)

        # Sharpen to enhance details
        upscaled = self._sharpen(upscaled, strength=1.2)

        return {
            "success": True,
            "upscaled": upscaled,
            "original_size": (width, height),
            "upscaled_size": (width * scale, height * scale),
            "scale_factor": scale,
        }

    def enhance_low_light(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhance low-light images."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Denoise (low-light images often have noise)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        return {
            "success": True,
            "enhanced": enhanced,
        }

    def enhance_contrast(
        self, image: np.ndarray, method: str = "clahe"
    ) -> Dict[str, Any]:
        """
        Enhance image contrast.

        Args:
            image: Input image (BGR)
            method: Enhancement method ("clahe", "histogram", "adaptive")

        Returns:
            Dictionary with contrast-enhanced image
        """
        if method == "clahe":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        elif method == "histogram":
            # Histogram Equalization
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            y_enhanced = cv2.equalizeHist(y)

            ycrcb_enhanced = cv2.merge([y_enhanced, cr, cb])
            enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

        elif method == "adaptive":
            # Adaptive Histogram Equalization
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced_gray = cv2.equalizeHist(gray)
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        else:
            return {"success": False, "error": f"Unknown method: {method}"}

        return {
            "success": True,
            "enhanced": enhanced,
            "method": method,
        }

    def _sharpen(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Sharpen image using unsharp masking."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        return sharpened

    def enhance_edges(self, image: np.ndarray, strength: float = 1.5) -> Dict[str, Any]:
        """Enhance edges in image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Convert edges to BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Blend with original image
        enhanced = cv2.addWeighted(image, 1.0, edges_bgr, strength * 0.3, 0)

        return {
            "success": True,
            "enhanced": enhanced,
            "strength": strength,
        }

    def remove_noise_and_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Remove noise and compression artifacts."""
        # Apply morphological operations to remove small artifacts
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Denoise
        cleaned = cv2.fastNlMeansDenoisingColored(cleaned, None, 10, 10, 7, 21)

        # Apply bilateral filter to preserve edges
        cleaned = cv2.bilateralFilter(cleaned, 9, 75, 75)

        return {
            "success": True,
            "cleaned": cleaned,
        }

    def enhance_for_forensics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Apply comprehensive forensic-grade enhancement pipeline.
        """
        # Step 1: Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Step 2: Enhance contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Step 3: Sharpen
        sharpened = self._sharpen(contrast_enhanced, strength=1.5)

        # Step 4: Enhance edges
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        final = cv2.addWeighted(sharpened, 1.0, edges_bgr, 0.3, 0)

        return {
            "success": True,
            "enhanced": final,
            "pipeline": ["denoise", "contrast_enhancement", "sharpening", "edge_enhancement"],
        }
