"""
Forensic analysis module with advanced image investigation tools.
Includes face reconstruction, EXIF analysis, object detection, layer analysis, etc.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import piexif
from PIL import Image
import json

logger = logging.getLogger(__name__)


class ForensicAnalyzer:
    """Main forensic analysis engine."""
    
    def __init__(self):
        self.face_reconstructor = FaceReconstructor()
        self.exif_analyzer = EXIFAnalyzer()
        self.object_detector = ObjectDetector()
        self.layer_analyzer = LayerAnalyzer()
        self.edge_enhancer = EdgeEnhancer()
    
    def full_analysis(self, image_path: str) -> Dict:
        """
        Perform full forensic analysis on an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with all analysis results
        """
        image = cv2.imread(image_path)
        
        return {
            "exif": self.exif_analyzer.analyze(image_path),
            "objects": self.object_detector.detect_all(image),
            "layers": self.layer_analyzer.decompose(image),
            "edges": self.edge_enhancer.analyze_edges(image),
            "reconstruction": self.face_reconstructor.assess_reconstructability(image)
        }


class FaceReconstructor:
    """Reconstruct faces from partial/masked/obscured images."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def reconstruct_from_masked(self, image: np.ndarray,
                                mask_region: Optional[np.ndarray] = None) -> Dict:
        """
        Reconstruct face from masked image using edge detection and pattern analysis.
        
        Args:
            image: Input image
            mask_region: Optional mask indicating obscured regions
        
        Returns:
            Dictionary with reconstruction results
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect face region
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {
                "success": False,
                "reason": "No face detected",
                "reconstructed": None
            }
        
        x, y, w, h = faces[0]
        face_region = image[y:y+h, x:x+w]
        
        # Analyze visible features
        visible_features = self._extract_visible_features(face_region)
        
        # Reconstruct obscured regions
        reconstructed = self._inpaint_obscured_regions(face_region, visible_features)
        
        # Enhance edges
        reconstructed = self._enhance_facial_edges(reconstructed)
        
        return {
            "success": True,
            "reconstructed": reconstructed,
            "visible_features": visible_features,
            "confidence": self._calculate_reconstruction_confidence(visible_features)
        }
    
    def _extract_visible_features(self, face: np.ndarray) -> Dict:
        """Extract visible facial features for reconstruction."""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        eyes = eye_cascade.detectMultiScale(gray)
        
        # Analyze skin tone
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        skin_pixels = face[skin_mask > 0]
        avg_skin_color = np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else None
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Analyze texture patterns
        texture_features = self._analyze_texture(gray)
        
        return {
            "eyes": eyes.tolist() if len(eyes) > 0 else [],
            "skin_color": avg_skin_color.tolist() if avg_skin_color is not None else None,
            "edge_map": edges,
            "texture": texture_features
        }
    
    def _analyze_texture(self, gray: np.ndarray) -> Dict:
        """Analyze texture patterns in grayscale image."""
        # Calculate local binary patterns
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i, j] = code
        
        return {
            "mean_texture": float(np.mean(lbp)),
            "std_texture": float(np.std(lbp)),
            "histogram": np.histogram(lbp, bins=256)[0].tolist()
        }
    
    def _inpaint_obscured_regions(self, face: np.ndarray, features: Dict) -> np.ndarray:
        """Inpaint obscured regions using available features."""
        # Create mask for obscured regions (simplified)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Inpaint using Navier-Stokes method
        inpainted = cv2.inpaint(face, mask, 3, cv2.INPAINT_NS)
        
        return inpainted
    
    def _enhance_facial_edges(self, face: np.ndarray) -> np.ndarray:
        """Enhance facial edges for better reconstruction."""
        # Convert to LAB color space
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _calculate_reconstruction_confidence(self, features: Dict) -> float:
        """Calculate confidence score for reconstruction."""
        score = 0.0
        
        if len(features["eyes"]) >= 2:
            score += 0.4
        elif len(features["eyes"]) >= 1:
            score += 0.2
        
        if features["skin_color"] is not None:
            score += 0.3
        
        if features["texture"]["std_texture"] > 10:
            score += 0.3
        
        return min(score, 1.0)
    
    def assess_reconstructability(self, image: np.ndarray) -> Dict:
        """Assess how reconstructable a face is from the image."""
        result = self.reconstruct_from_masked(image)
        return {
            "reconstructable": result["success"],
            "confidence": result.get("confidence", 0.0),
            "visible_features_count": len(result.get("visible_features", {}).get("eyes", [])),
            "recommendation": self._get_reconstruction_recommendation(result.get("confidence", 0.0))
        }
    
    def _get_reconstruction_recommendation(self, confidence: float) -> str:
        """Get recommendation based on confidence."""
        if confidence >= 0.7:
            return "High confidence - good reconstruction possible"
        elif confidence >= 0.4:
            return "Medium confidence - partial reconstruction possible"
        else:
            return "Low confidence - limited reconstruction possible"


class EXIFAnalyzer:
    """Analyze and recover EXIF metadata from images."""
    
    def analyze(self, image_path: str) -> Dict:
        """
        Extract and analyze EXIF data.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with EXIF data and analysis
        """
        try:
            img = Image.open(image_path)
            exif_dict = piexif.load(img.info.get("exif", b""))
            
            # Extract key metadata
            metadata = {
                "camera": self._extract_camera_info(exif_dict),
                "location": self._extract_location(exif_dict),
                "datetime": self._extract_datetime(exif_dict),
                "settings": self._extract_camera_settings(exif_dict),
                "software": self._extract_software_info(exif_dict),
                "raw_exif": self._serialize_exif(exif_dict)
            }
            
            # Analyze for manipulation
            metadata["manipulation_indicators"] = self._detect_manipulation(exif_dict)
            
            return metadata
        
        except Exception as e:
            logger.error(f"EXIF analysis error: {e}")
            return {
                "error": str(e),
                "camera": None,
                "location": None,
                "datetime": None
            }
    
    def _extract_camera_info(self, exif_dict: Dict) -> Optional[Dict]:
        """Extract camera information."""
        try:
            ifd0 = exif_dict.get("0th", {})
            exif = exif_dict.get("Exif", {})
            
            return {
                "make": ifd0.get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore"),
                "model": ifd0.get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore"),
                "lens": exif.get(piexif.ExifIFD.LensModel, b"").decode("utf-8", errors="ignore")
            }
        except:
            return None
    
    def _extract_location(self, exif_dict: Dict) -> Optional[Dict]:
        """Extract GPS location."""
        try:
            gps = exif_dict.get("GPS", {})
            if not gps:
                return None
            
            lat = self._convert_to_degrees(gps.get(piexif.GPSIFD.GPSLatitude, []))
            lon = self._convert_to_degrees(gps.get(piexif.GPSIFD.GPSLongitude, []))
            
            lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode()
            lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode()
            
            if lat_ref == "S":
                lat = -lat
            if lon_ref == "W":
                lon = -lon
            
            return {
                "latitude": lat,
                "longitude": lon,
                "altitude": gps.get(piexif.GPSIFD.GPSAltitude, None)
            }
        except:
            return None
    
    def _convert_to_degrees(self, value):
        """Convert GPS coordinates to degrees."""
        if not value or len(value) != 3:
            return 0
        d = value[0][0] / value[0][1]
        m = value[1][0] / value[1][1]
        s = value[2][0] / value[2][1]
        return d + (m / 60.0) + (s / 3600.0)
    
    def _extract_datetime(self, exif_dict: Dict) -> Optional[str]:
        """Extract datetime information."""
        try:
            exif = exif_dict.get("Exif", {})
            dt = exif.get(piexif.ExifIFD.DateTimeOriginal, b"").decode("utf-8", errors="ignore")
            return dt if dt else None
        except:
            return None
    
    def _extract_camera_settings(self, exif_dict: Dict) -> Dict:
        """Extract camera settings."""
        try:
            exif = exif_dict.get("Exif", {})
            return {
                "iso": exif.get(piexif.ExifIFD.ISOSpeedRatings, None),
                "exposure_time": exif.get(piexif.ExifIFD.ExposureTime, None),
                "f_number": exif.get(piexif.ExifIFD.FNumber, None),
                "focal_length": exif.get(piexif.ExifIFD.FocalLength, None),
                "flash": exif.get(piexif.ExifIFD.Flash, None)
            }
        except:
            return {}
    
    def _extract_software_info(self, exif_dict: Dict) -> Optional[str]:
        """Extract software/editing information."""
        try:
            ifd0 = exif_dict.get("0th", {})
            software = ifd0.get(piexif.ImageIFD.Software, b"").decode("utf-8", errors="ignore")
            return software if software else None
        except:
            return None
    
    def _serialize_exif(self, exif_dict: Dict) -> Dict:
        """Serialize EXIF dict to JSON-compatible format."""
        result = {}
        for ifd_name, ifd_data in exif_dict.items():
            result[ifd_name] = {}
            for tag, value in ifd_data.items():
                try:
                    if isinstance(value, bytes):
                        result[ifd_name][str(tag)] = value.decode("utf-8", errors="ignore")
                    else:
                        result[ifd_name][str(tag)] = str(value)
                except:
                    pass
        return result
    
    def _detect_manipulation(self, exif_dict: Dict) -> List[str]:
        """Detect indicators of image manipulation."""
        indicators = []
        
        # Check for editing software
        software = self._extract_software_info(exif_dict)
        if software and any(editor in software.lower() for editor in ["photoshop", "gimp", "lightroom"]):
            indicators.append(f"Edited with: {software}")
        
        # Check for missing expected EXIF data
        if not self._extract_camera_info(exif_dict):
            indicators.append("Missing camera information")
        
        if not self._extract_datetime(exif_dict):
            indicators.append("Missing datetime information")
        
        return indicators


class ObjectDetector:
    """Detect objects, clothing tags, tattoos, jewelry, etc."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_all(self, image: np.ndarray) -> Dict:
        """Detect all objects of interest."""
        return {
            "glasses": self.detect_glasses(image),
            "jewelry": self.detect_jewelry(image),
            "clothing_tags": self.detect_clothing_tags(image),
            "tattoos": self.detect_tattoos(image),
            "text": self.detect_text(image)
        }
    
    def detect_glasses(self, image: np.ndarray) -> Dict:
        """Detect eyeglasses."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray)
        
        glasses_detected = False
        confidence = 0.0
        
        if len(eyes) >= 2:
            # Analyze region around eyes for glass frames
            for (ex, ey, ew, eh) in eyes:
                eye_region = gray[ey:ey+eh, ex:ex+ew]
                edges = cv2.Canny(eye_region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                if edge_density > 0.1:
                    glasses_detected = True
                    confidence = min(edge_density * 5, 1.0)
                    break
        
        return {
            "detected": glasses_detected,
            "confidence": confidence,
            "count": len(eyes)
        }
    
    def detect_jewelry(self, image: np.ndarray) -> Dict:
        """Detect jewelry (simplified - looks for metallic/shiny objects)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect metallic/shiny regions
        lower_metal = np.array([0, 0, 200])
        upper_metal = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_metal, upper_metal)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        jewelry_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 5000:  # Filter by size
                x, y, w, h = cv2.boundingRect(cnt)
                jewelry_regions.append([int(x), int(y), int(w), int(h)])
        
        return {
            "detected": len(jewelry_regions) > 0,
            "count": len(jewelry_regions),
            "regions": jewelry_regions
        }
    
    def detect_clothing_tags(self, image: np.ndarray) -> Dict:
        """Detect clothing tags (looks for small text regions)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect text-like regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tag_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Tags are typically rectangular with specific aspect ratio
            if 0.5 < aspect_ratio < 3.0 and 50 < w * h < 2000:
                tag_regions.append([int(x), int(y), int(w), int(h)])
        
        return {
            "detected": len(tag_regions) > 0,
            "count": len(tag_regions),
            "regions": tag_regions
        }
    
    def detect_tattoos(self, image: np.ndarray) -> Dict:
        """Detect tattoos (looks for high-contrast skin markings)."""
        # Convert to HSV and extract skin regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply mask to get skin regions
        skin_region = cv2.bitwise_and(image, image, mask=skin_mask)
        gray_skin = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
        
        # Detect high-contrast regions (potential tattoos)
        edges = cv2.Canny(gray_skin, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tattoo_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Minimum size for tattoo
                x, y, w, h = cv2.boundingRect(cnt)
                tattoo_regions.append([int(x), int(y), int(w), int(h)])
        
        return {
            "detected": len(tattoo_regions) > 0,
            "count": len(tattoo_regions),
            "regions": tattoo_regions
        }
    
    def detect_text(self, image: np.ndarray) -> Dict:
        """Detect text in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        for region in regions:
            if len(region) > 10:
                x, y, w, h = cv2.boundingRect(region)
                text_regions.append([int(x), int(y), int(w), int(h)])
        
        return {
            "detected": len(text_regions) > 0,
            "count": len(text_regions),
            "regions": text_regions[:50]  # Limit to first 50
        }


class LayerAnalyzer:
    """Decompose images into layers and analyze progressively."""
    
    def decompose(self, image: np.ndarray, num_layers: int = 5) -> List[np.ndarray]:
        """
        Decompose image into layers.
        
        Args:
            image: Input image
            num_layers: Number of layers to extract
        
        Returns:
            List of layer images
        """
        layers = []
        
        # Frequency-based decomposition
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for i in range(num_layers):
            # Apply Gaussian blur with increasing sigma
            sigma = 2 ** i
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            
            # Extract detail layer
            if i == 0:
                layer = gray - blurred
            else:
                prev_blurred = cv2.GaussianBlur(gray, (0, 0), 2 ** (i - 1))
                layer = prev_blurred - blurred
            
            # Normalize
            layer = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            layers.append(layer)
        
        return layers
    
    def progressive_removal(self, image: np.ndarray, layer_index: int,
                           removal_strength: float = 0.5) -> np.ndarray:
        """
        Progressively remove a layer from the image.
        
        Args:
            image: Input image
            layer_index: Index of layer to remove
            removal_strength: Strength of removal (0-1)
        
        Returns:
            Image with layer partially removed
        """
        layers = self.decompose(image)
        
        if layer_index >= len(layers):
            return image
        
        # Remove layer
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        layer = layers[layer_index]
        
        result = gray - (layer * removal_strength).astype(np.uint8)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result


class EdgeEnhancer:
    """Enhance edges using lighting and makeup patterns."""
    
    def analyze_edges(self, image: np.ndarray) -> Dict:
        """Analyze edge patterns in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection methods
        canny = cv2.Canny(gray, 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return {
            "edge_density": float(np.sum(canny > 0) / canny.size),
            "edge_strength": float(np.mean(sobel)),
            "edge_map": canny
        }
    
    def enhance_facial_edges(self, image: np.ndarray,
                            strength: float = 1.0) -> np.ndarray:
        """
        Enhance facial edges using lighting and makeup patterns.
        
        Args:
            image: Input image
            strength: Enhancement strength (0-2)
        
        Returns:
            Edge-enhanced image
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance L channel
        clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Detect edges
        edges = cv2.Canny(l, 50, 150)
        
        # Enhance edges in L channel
        l_enhanced = cv2.add(l_enhanced, (edges * strength * 0.5).astype(np.uint8))
        
        # Merge and convert back
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def zoom_and_enhance(self, image: np.ndarray, region: Tuple[int, int, int, int],
                        scale: float = 2.0) -> np.ndarray:
        """
        Zoom into a region and enhance details.
        
        Args:
            image: Input image
            region: Region to zoom (x, y, w, h)
            scale: Zoom scale factor
        
        Returns:
            Zoomed and enhanced region
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        
        # Upscale
        new_size = (int(w * scale), int(h * scale))
        zoomed = cv2.resize(roi, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Enhance
        enhanced = self.enhance_facial_edges(zoomed, strength=1.5)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
