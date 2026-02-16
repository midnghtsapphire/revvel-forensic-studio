"""
Face detection and landmark extraction module.
Supports multiple backends: dlib, MediaPipe, OpenCV.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")


class FaceDetector:
    """Multi-backend face detector with landmark extraction."""
    
    def __init__(self, backend: str = "auto"):
        """
        Initialize face detector.
        
        Args:
            backend: "dlib", "mediapipe", "opencv", or "auto"
        """
        self.backend = backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the selected backend."""
        if self.backend == "auto":
            if MEDIAPIPE_AVAILABLE:
                self.backend = "mediapipe"
            elif DLIB_AVAILABLE:
                self.backend = "dlib"
            else:
                self.backend = "opencv"
        
        if self.backend == "dlib" and DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        elif self.backend == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                min_detection_confidence=0.5
            )
        
        elif self.backend == "opencv":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        
        logger.info(f"Face detector initialized with backend: {self.backend}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of face dictionaries with bbox and landmarks
        """
        if self.backend == "dlib":
            return self._detect_dlib(image)
        elif self.backend == "mediapipe":
            return self._detect_mediapipe(image)
        elif self.backend == "opencv":
            return self._detect_opencv(image)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _detect_dlib(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        
        results = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Get landmarks
            shape = self.predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            results.append({
                "bbox": [x, y, w, h],
                "landmarks": landmarks,
                "confidence": 1.0,
                "backend": "dlib"
            })
        
        return results
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_detection = self.face_detection.process(rgb)
        results_mesh = self.face_mesh.process(rgb)
        
        faces = []
        
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = np.array([
                    [int(lm.x * w), int(lm.y * w)]
                    for lm in face_landmarks.landmark
                ])
                
                # Calculate bounding box from landmarks
                x_min = landmarks[:, 0].min()
                y_min = landmarks[:, 1].min()
                x_max = landmarks[:, 0].max()
                y_max = landmarks[:, 1].max()
                
                faces.append({
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "landmarks": landmarks,
                    "confidence": 0.9,
                    "backend": "mediapipe"
                })
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            # Detect eyes as simple landmarks
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            landmarks = []
            for (ex, ey, ew, eh) in eyes:
                landmarks.append([x + ex + ew//2, y + ey + eh//2])
            
            results.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "landmarks": np.array(landmarks) if landmarks else np.array([]),
                "confidence": 0.7,
                "backend": "opencv"
            })
        
        return results
    
    def get_face_region(self, image: np.ndarray, bbox: List[int],
                        padding: float = 0.2) -> np.ndarray:
        """
        Extract face region with padding.
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, w, h]
            padding: Padding ratio (0.2 = 20% padding)
        
        Returns:
            Cropped face region
        """
        x, y, w, h = bbox
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray,
                   output_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Align face using landmarks.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            output_size: Output image size
        
        Returns:
            Aligned face image
        """
        if len(landmarks) < 2:
            # No landmarks, just resize
            return cv2.resize(image, output_size)
        
        # Use eye positions for alignment if available
        if len(landmarks) >= 36:  # dlib 68-point landmarks
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
        else:
            # Use first two landmarks as eyes
            left_eye = landmarks[0]
            right_eye = landmarks[1]
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get rotation matrix
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Rotate image
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_CUBIC)
        
        # Resize to output size
        aligned = cv2.resize(aligned, output_size)
        
        return aligned


class MaskDetector:
    """Detect if a face is wearing a mask based on facial feature analysis."""
    
    def __init__(self):
        self.face_detector = FaceDetector()
    
    def detect_mask(self, image: np.ndarray) -> Dict:
        """
        Detect if face is wearing a mask.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with mask detection results
        """
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            return {
                "mask_detected": False,
                "confidence": 0.0,
                "reason": "No face detected"
            }
        
        face = faces[0]
        landmarks = face["landmarks"]
        
        if len(landmarks) < 10:
            return {
                "mask_detected": False,
                "confidence": 0.0,
                "reason": "Insufficient landmarks"
            }
        
        # Analyze nose and mouth region
        bbox = face["bbox"]
        x, y, w, h = bbox
        
        # Extract lower face region
        lower_face = image[y + h//2:y + h, x:x + w]
        
        # Analyze texture and edge patterns
        gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Check for uniform texture (mask characteristic)
        std_dev = np.std(gray)
        
        # Analyze color distribution
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv[:, :, 1])  # Saturation variance
        
        # Scoring
        mask_score = 0.0
        reasons = []
        
        if edge_density < 0.05:
            mask_score += 0.3
            reasons.append("Low edge density in lower face")
        
        if std_dev < 20:
            mask_score += 0.3
            reasons.append("Uniform texture in lower face")
        
        if color_variance < 100:
            mask_score += 0.2
            reasons.append("Low color variance")
        
        # Check for visible mouth/nose landmarks
        if len(landmarks) >= 68:
            nose_tip = landmarks[30]
            mouth_center = landmarks[62]
            
            # If nose/mouth landmarks are obscured or flattened
            nose_mouth_dist = np.linalg.norm(nose_tip - mouth_center)
            if nose_mouth_dist < h * 0.1:
                mask_score += 0.2
                reasons.append("Flattened nose-mouth region")
        
        mask_detected = mask_score > 0.5
        
        return {
            "mask_detected": mask_detected,
            "confidence": min(mask_score, 1.0),
            "reasons": reasons,
            "metrics": {
                "edge_density": float(edge_density),
                "texture_std": float(std_dev),
                "color_variance": float(color_variance)
            }
        }


def load_image(path: str) -> np.ndarray:
    """Load image from file."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def save_image(image: np.ndarray, path: str):
    """Save image to file."""
    cv2.imwrite(path, image)
