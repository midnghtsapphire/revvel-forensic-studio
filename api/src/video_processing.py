"""
Revvel Forensic Studio — Video Frame Extraction
Extract frames from video with AI-powered scene detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import timedelta
import json


class VideoFrameExtractor:
    """Extract frames from video files with scene detection."""

    def __init__(self):
        self.scene_threshold = 30.0  # Threshold for scene change detection

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        mode: str = "interval",
        interval: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            mode: Extraction mode ("interval", "scene_change", "keyframes", "all")
            interval: Seconds between frames (for interval mode)
            max_frames: Maximum number of frames to extract

        Returns:
            Dictionary with extraction results
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Failed to open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        result = {
            "success": True,
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration,
            "mode": mode,
            "extracted_frames": [],
        }

        if mode == "interval":
            result["extracted_frames"] = self._extract_by_interval(
                cap, output_dir, interval, fps, max_frames
            )
        elif mode == "scene_change":
            result["extracted_frames"] = self._extract_by_scene_change(
                cap, output_dir, max_frames
            )
        elif mode == "keyframes":
            result["extracted_frames"] = self._extract_keyframes(
                cap, output_dir, max_frames
            )
        elif mode == "all":
            result["extracted_frames"] = self._extract_all_frames(
                cap, output_dir, max_frames
            )

        cap.release()

        # Save metadata
        metadata_path = os.path.join(output_dir, "extraction_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _extract_by_interval(
        self,
        cap: cv2.VideoCapture,
        output_dir: str,
        interval: float,
        fps: float,
        max_frames: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Extract frames at regular intervals."""
        frames = []
        frame_interval = int(interval * fps)
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                filename = f"frame_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, frame)

                frames.append({
                    "frame_number": frame_count,
                    "timestamp": timestamp,
                    "filename": filename,
                    "path": output_path,
                })

                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        return frames

    def _extract_by_scene_change(
        self,
        cap: cv2.VideoCapture,
        output_dir: str,
        max_frames: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Extract frames at scene changes using histogram comparison."""
        frames = []
        prev_frame = None
        frame_count = 0
        extracted_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # Calculate histogram difference
                hist_diff = self._calculate_histogram_difference(prev_frame, frame)

                if hist_diff > self.scene_threshold:
                    timestamp = frame_count / fps
                    filename = f"scene_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, frame)

                    frames.append({
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "filename": filename,
                        "path": output_path,
                        "scene_change_score": hist_diff,
                    })

                    extracted_count += 1
                    if max_frames and extracted_count >= max_frames:
                        break

            prev_frame = frame.copy()
            frame_count += 1

        return frames

    def _extract_keyframes(
        self,
        cap: cv2.VideoCapture,
        output_dir: str,
        max_frames: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Extract keyframes (frames with significant motion or changes)."""
        frames = []
        prev_frame = None
        frame_count = 0
        extracted_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # Calculate motion score
                motion_score = self._calculate_motion_score(prev_frame, frame)

                # Extract if motion score is high (keyframe)
                if motion_score > 15.0:
                    timestamp = frame_count / fps
                    filename = f"keyframe_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, frame)

                    frames.append({
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "filename": filename,
                        "path": output_path,
                        "motion_score": motion_score,
                    })

                    extracted_count += 1
                    if max_frames and extracted_count >= max_frames:
                        break

            prev_frame = frame.copy()
            frame_count += 1

        return frames

    def _extract_all_frames(
        self,
        cap: cv2.VideoCapture,
        output_dir: str,
        max_frames: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Extract all frames from video."""
        frames = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps
            filename = f"frame_{frame_count:06d}_t{timestamp:.2f}s.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)

            frames.append({
                "frame_number": frame_count,
                "timestamp": timestamp,
                "filename": filename,
                "path": output_path,
            })

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        return frames

    def _calculate_histogram_difference(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """Calculate histogram difference between two frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Calculate correlation (lower = more different)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Convert to difference score (0-100, higher = more different)
        difference = (1 - correlation) * 100

        return difference

    def _calculate_motion_score(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """Calculate motion score between two frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Calculate mean difference
        motion_score = np.mean(diff)

        return motion_score

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Failed to open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "success": True,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "duration_formatted": str(timedelta(seconds=int(duration))),
        }

    def extract_frame_at_timestamp(
        self, video_path: str, timestamp: float, output_path: str
    ) -> bool:
        """Extract a single frame at a specific timestamp."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(output_path, frame)
            cap.release()
            return True

        cap.release()
        return False
