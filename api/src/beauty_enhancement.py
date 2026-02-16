"""
Beauty enhancement module - YouCam Perfect style features.
Includes makeup, skin smoothing, face reshaping, and batch processing.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from .face_detection import FaceDetector

logger = logging.getLogger(__name__)


class BeautyEnhancer:
    """Main beauty enhancement engine."""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.skin_smoother = SkinSmoother()
        self.makeup_artist = MakeupArtist()
        self.face_reshaper = FaceReshaper()
    
    def enhance(self, image: np.ndarray, preset: str = "natural",
                custom_params: Optional[Dict] = None) -> np.ndarray:
        """
        Apply beauty enhancements to image.
        
        Args:
            image: Input image
            preset: Enhancement preset ("natural", "glamour", "dramatic", "subtle")
            custom_params: Custom enhancement parameters
        
        Returns:
            Enhanced image
        """
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            logger.warning("No faces detected, returning original image")
            return image
        
        result = image.copy()
        params = self._get_preset_params(preset)
        
        if custom_params:
            params.update(custom_params)
        
        for face in faces:
            result = self._enhance_face(result, face, params)
        
        return result
    
    def _get_preset_params(self, preset: str) -> Dict:
        """Get enhancement parameters for preset."""
        presets = {
            "natural": {
                "skin_smooth": 0.3,
                "blemish_removal": 0.5,
                "brightness": 1.05,
                "contour_strength": 0.2,
                "highlight_strength": 0.3,
                "lipstick_opacity": 0.0,
                "eyeshadow_opacity": 0.0,
                "teeth_whitening": 0.3,
                "eye_enhancement": 0.2,
            },
            "glamour": {
                "skin_smooth": 0.6,
                "blemish_removal": 0.8,
                "brightness": 1.1,
                "contour_strength": 0.5,
                "highlight_strength": 0.6,
                "lipstick_opacity": 0.5,
                "eyeshadow_opacity": 0.4,
                "teeth_whitening": 0.6,
                "eye_enhancement": 0.5,
            },
            "dramatic": {
                "skin_smooth": 0.8,
                "blemish_removal": 1.0,
                "brightness": 1.15,
                "contour_strength": 0.8,
                "highlight_strength": 0.8,
                "lipstick_opacity": 0.8,
                "eyeshadow_opacity": 0.7,
                "teeth_whitening": 0.8,
                "eye_enhancement": 0.7,
            },
            "subtle": {
                "skin_smooth": 0.2,
                "blemish_removal": 0.3,
                "brightness": 1.02,
                "contour_strength": 0.1,
                "highlight_strength": 0.2,
                "lipstick_opacity": 0.0,
                "eyeshadow_opacity": 0.0,
                "teeth_whitening": 0.2,
                "eye_enhancement": 0.1,
            },
        }
        return presets.get(preset, presets["natural"])
    
    def _enhance_face(self, image: np.ndarray, face: Dict, params: Dict) -> np.ndarray:
        """Apply enhancements to a single face."""
        result = image.copy()
        bbox = face["bbox"]
        landmarks = face["landmarks"]
        
        # Get face region
        face_region = self.face_detector.get_face_region(result, bbox)
        
        # Apply enhancements
        if params.get("skin_smooth", 0) > 0:
            face_region = self.skin_smoother.smooth_skin(
                face_region, strength=params["skin_smooth"]
            )
        
        if params.get("blemish_removal", 0) > 0:
            face_region = self.skin_smoother.remove_blemishes(
                face_region, strength=params["blemish_removal"]
            )
        
        if params.get("brightness", 1.0) != 1.0:
            face_region = self._adjust_brightness(face_region, params["brightness"])
        
        # Apply makeup
        if params.get("contour_strength", 0) > 0:
            face_region = self.makeup_artist.apply_contour(
                face_region, landmarks, strength=params["contour_strength"]
            )
        
        if params.get("highlight_strength", 0) > 0:
            face_region = self.makeup_artist.apply_highlight(
                face_region, landmarks, strength=params["highlight_strength"]
            )
        
        if params.get("lipstick_opacity", 0) > 0:
            face_region = self.makeup_artist.apply_lipstick(
                face_region, landmarks, opacity=params["lipstick_opacity"]
            )
        
        if params.get("eyeshadow_opacity", 0) > 0:
            face_region = self.makeup_artist.apply_eyeshadow(
                face_region, landmarks, opacity=params["eyeshadow_opacity"]
            )
        
        # Place enhanced face back
        x, y, w, h = bbox
        pad = int(w * 0.2)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(result.shape[1], x + w + pad), min(result.shape[0], y + h + pad)
        
        result[y1:y2, x1:x2] = face_region
        
        return result
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def batch_enhance(self, images: List[np.ndarray], preset: str = "natural",
                      progress_callback=None) -> List[np.ndarray]:
        """
        Batch process multiple images.
        
        Args:
            images: List of input images
            preset: Enhancement preset
            progress_callback: Optional callback(current, total)
        
        Returns:
            List of enhanced images
        """
        results = []
        total = len(images)
        
        for i, image in enumerate(images):
            enhanced = self.enhance(image, preset)
            results.append(enhanced)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results


class SkinSmoother:
    """Skin smoothing and blemish removal."""
    
    def smooth_skin(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Apply skin smoothing using bilateral filter.
        
        Args:
            image: Input image
            strength: Smoothing strength (0-1)
        
        Returns:
            Smoothed image
        """
        if strength <= 0:
            return image
        
        # Convert strength to filter parameters
        d = int(5 + strength * 10)
        sigma_color = 50 + strength * 50
        sigma_space = 50 + strength * 50
        
        # Apply bilateral filter
        smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Blend with original
        result = cv2.addWeighted(image, 1 - strength, smoothed, strength, 0)
        
        return result
    
    def remove_blemishes(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Remove blemishes using inpainting.
        
        Args:
            image: Input image
            strength: Removal strength (0-1)
        
        Returns:
            Image with blemishes removed
        """
        if strength <= 0:
            return image
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect dark spots (blemishes)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create mask for inpainting
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Inpaint
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # Blend based on strength
        result = cv2.addWeighted(image, 1 - strength, inpainted, strength, 0)
        
        return result


class MakeupArtist:
    """Apply virtual makeup."""
    
    def apply_contour(self, image: np.ndarray, landmarks: np.ndarray,
                      strength: float = 0.5, color: Tuple[int, int, int] = (139, 115, 85)) -> np.ndarray:
        """Apply contour makeup."""
        if len(landmarks) < 10:
            return image
        
        result = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Define contour regions (cheekbones, jawline, forehead)
        # This is simplified - real implementation would use specific landmark indices
        h, w = image.shape[:2]
        
        # Cheekbone contour
        if len(landmarks) >= 17:
            cheek_points = landmarks[1:16].astype(np.int32)
            cv2.fillPoly(mask, [cheek_points], 255)
        
        # Apply color
        mask_blurred = cv2.GaussianBlur(mask, (51, 51), 0)
        mask_normalized = mask_blurred.astype(np.float32) / 255.0 * strength
        
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] * (1 - mask_normalized * 0.3) + color[c] * mask_normalized * 0.3,
                0, 255
            ).astype(np.uint8)
        
        return result
    
    def apply_highlight(self, image: np.ndarray, landmarks: np.ndarray,
                        strength: float = 0.5) -> np.ndarray:
        """Apply highlight makeup."""
        if len(landmarks) < 10:
            return image
        
        result = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Highlight regions: nose bridge, cheekbones, brow bone
        if len(landmarks) >= 30:
            # Nose bridge
            nose_points = landmarks[27:31].astype(np.int32)
            for pt in nose_points:
                cv2.circle(mask, tuple(pt), 5, 255, -1)
        
        # Apply highlight
        mask_blurred = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_normalized = mask_blurred.astype(np.float32) / 255.0 * strength
        
        # Brighten highlighted areas
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + mask_normalized * 50, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def apply_lipstick(self, image: np.ndarray, landmarks: np.ndarray,
                       opacity: float = 0.5, color: Tuple[int, int, int] = (0, 0, 180)) -> np.ndarray:
        """Apply lipstick."""
        if len(landmarks) < 48:
            return image
        
        result = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Lip region (landmarks 48-68 in dlib 68-point model)
        if len(landmarks) >= 68:
            lip_points = landmarks[48:68].astype(np.int32)
            cv2.fillPoly(mask, [lip_points], 255)
        
        # Apply color
        mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0)
        mask_normalized = mask_blurred.astype(np.float32) / 255.0 * opacity
        
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] * (1 - mask_normalized) + color[c] * mask_normalized,
                0, 255
            ).astype(np.uint8)
        
        return result
    
    def apply_eyeshadow(self, image: np.ndarray, landmarks: np.ndarray,
                        opacity: float = 0.5, color: Tuple[int, int, int] = (139, 90, 43)) -> np.ndarray:
        """Apply eyeshadow."""
        if len(landmarks) < 36:
            return image
        
        result = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Eye regions (landmarks 36-48 in dlib 68-point model)
        if len(landmarks) >= 48:
            left_eye = landmarks[36:42].astype(np.int32)
            right_eye = landmarks[42:48].astype(np.int32)
            
            cv2.fillPoly(mask, [left_eye], 255)
            cv2.fillPoly(mask, [right_eye], 255)
        
        # Apply color
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_normalized = mask_blurred.astype(np.float32) / 255.0 * opacity
        
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] * (1 - mask_normalized * 0.5) + color[c] * mask_normalized * 0.5,
                0, 255
            ).astype(np.uint8)
        
        return result


class FaceReshaper:
    """Face and body reshaping tools."""
    
    def reshape_face(self, image: np.ndarray, landmarks: np.ndarray,
                     chin: float = 0.0, cheeks: float = 0.0,
                     nose: float = 0.0, jawline: float = 0.0) -> np.ndarray:
        """
        Reshape facial features.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            chin: Chin adjustment (-1 to 1, negative = smaller)
            cheeks: Cheek adjustment (-1 to 1, negative = slimmer)
            nose: Nose adjustment (-1 to 1, negative = narrower)
            jawline: Jawline adjustment (-1 to 1, negative = sharper)
        
        Returns:
            Reshaped image
        """
        if len(landmarks) < 17:
            return image
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Create displacement map
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        
        # Apply transformations based on landmarks
        if chin != 0 and len(landmarks) >= 9:
            chin_point = landmarks[8]
            radius = h * 0.15
            self._apply_local_warp(map_x, map_y, chin_point, radius, chin * 10, 0)
        
        if cheeks != 0 and len(landmarks) >= 17:
            left_cheek = landmarks[3]
            right_cheek = landmarks[13]
            radius = w * 0.1
            self._apply_local_warp(map_x, map_y, left_cheek, radius, cheeks * 5, 0)
            self._apply_local_warp(map_x, map_y, right_cheek, radius, -cheeks * 5, 0)
        
        # Apply remap
        result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR)
        
        return result
    
    def _apply_local_warp(self, map_x: np.ndarray, map_y: np.ndarray,
                          center: np.ndarray, radius: float,
                          dx: float, dy: float):
        """Apply local warp to displacement map."""
        cx, cy = center
        h, w = map_x.shape
        
        # Create distance map
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        # Apply smooth falloff
        mask = np.clip(1 - dist / radius, 0, 1)
        mask = mask ** 2  # Smooth falloff
        
        # Apply displacement
        map_x += mask * dx
        map_y += mask * dy


class BatchProcessor:
    """Batch image processing with progress tracking."""
    
    def __init__(self, enhancer: BeautyEnhancer):
        self.enhancer = enhancer
    
    def process_batch(self, input_paths: List[str], output_dir: str,
                      preset: str = "natural", max_count: Optional[int] = None,
                      progress_callback=None) -> List[str]:
        """
        Process batch of images.
        
        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            preset: Enhancement preset
            max_count: Maximum number of images to process
            progress_callback: Optional callback(current, total, path)
        
        Returns:
            List of output paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if max_count:
            input_paths = input_paths[:max_count]
        
        output_paths = []
        total = len(input_paths)
        
        for i, input_path in enumerate(input_paths):
            try:
                # Load image
                image = cv2.imread(input_path)
                if image is None:
                    logger.warning(f"Could not load: {input_path}")
                    continue
                
                # Enhance
                enhanced = self.enhancer.enhance(image, preset)
                
                # Save
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, f"enhanced_{filename}")
                cv2.imwrite(output_path, enhanced)
                output_paths.append(output_path)
                
                if progress_callback:
                    progress_callback(i + 1, total, output_path)
            
            except Exception as e:
                logger.error(f"Error processing {input_path}: {e}")
        
        return output_paths
