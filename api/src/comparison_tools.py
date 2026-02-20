"""
Revvel Forensic Studio — Comparison Tools
Side-by-side image comparison with difference highlighting.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from skimage.metrics import structural_similarity as ssim


class ComparisonTools:
    """Tools for comparing and analyzing multiple images."""

    def side_by_side_comparison(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        labels: Tuple[str, str] = ("Image 1", "Image 2"),
        show_difference: bool = True,
    ) -> Dict[str, Any]:
        """
        Create side-by-side comparison of two images.

        Args:
            image1: First image (BGR)
            image2: Second image (BGR)
            labels: Tuple of labels for images
            show_difference: Include difference map

        Returns:
            Dictionary with comparison results
        """
        # Resize images to same size
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        target_height = max(h1, h2)
        target_width = max(w1, w2)

        img1_resized = cv2.resize(image1, (target_width, target_height))
        img2_resized = cv2.resize(image2, (target_width, target_height))

        # Create side-by-side image
        side_by_side = np.hstack([img1_resized, img2_resized])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            side_by_side,
            labels[0],
            (10, 30),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            side_by_side,
            labels[1],
            (target_width + 10, 30),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        result = {
            "success": True,
            "side_by_side": side_by_side,
            "image1_size": (w1, h1),
            "image2_size": (w2, h2),
        }

        if show_difference:
            difference_map = self.create_difference_map(img1_resized, img2_resized)
            result["difference_map"] = difference_map["difference_image"]
            result["similarity_score"] = difference_map["similarity_score"]

        return result

    def create_difference_map(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create difference map highlighting changes between two images.

        Args:
            image1: First image (BGR)
            image2: Second image (BGR)

        Returns:
            Dictionary with difference map and metrics
        """
        # Ensure same size
        if image1.shape != image2.shape:
            h, w = image1.shape[:2]
            image2 = cv2.resize(image2, (w, h))

        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        similarity_score, diff = ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype(np.uint8)

        # Threshold difference
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)

        # Create colored difference map
        diff_colored = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)

        # Blend with original image
        blended = cv2.addWeighted(image1, 0.7, diff_colored, 0.3, 0)

        return {
            "success": True,
            "difference_image": blended,
            "difference_mask": thresh,
            "similarity_score": float(similarity_score),
            "difference_percentage": float((1 - similarity_score) * 100),
        }

    def highlight_differences(
        self, image1: np.ndarray, image2: np.ndarray, threshold: int = 30
    ) -> Dict[str, Any]:
        """
        Highlight specific differences between two images.

        Args:
            image1: First image (BGR)
            image2: Second image (BGR)
            threshold: Difference threshold (0-255)

        Returns:
            Dictionary with highlighted differences
        """
        # Ensure same size
        if image1.shape != image2.shape:
            h, w = image1.shape[:2]
            image2 = cv2.resize(image2, (w, h))

        # Calculate absolute difference
        diff = cv2.absdiff(image1, image2)

        # Convert to grayscale
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold to find significant differences
        _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

        # Find contours of differences
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around differences
        result_image = image1.copy()
        difference_regions = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                difference_regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "area": int(area)})

        return {
            "success": True,
            "highlighted_image": result_image,
            "difference_mask": mask,
            "num_differences": len(difference_regions),
            "difference_regions": difference_regions,
        }

    def multi_image_comparison(
        self, images: List[np.ndarray], labels: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple images in a grid layout.

        Args:
            images: List of images (BGR)
            labels: List of labels for each image

        Returns:
            Dictionary with grid comparison
        """
        if len(images) != len(labels):
            return {"success": False, "error": "Number of images and labels must match"}

        # Determine grid size
        num_images = len(images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

        # Resize all images to same size
        target_height = max(img.shape[0] for img in images)
        target_width = max(img.shape[1] for img in images)

        resized_images = [
            cv2.resize(img, (target_width, target_height)) for img in images
        ]

        # Add labels to images
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (img, label) in enumerate(zip(resized_images, labels)):
            cv2.putText(img, label, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Create grid
        grid_rows = []
        for row in range(rows):
            row_images = []
            for col in range(cols):
                idx = row * cols + col
                if idx < num_images:
                    row_images.append(resized_images[idx])
                else:
                    # Fill empty slots with black image
                    row_images.append(
                        np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    )
            grid_rows.append(np.hstack(row_images))

        grid = np.vstack(grid_rows)

        return {
            "success": True,
            "grid_image": grid,
            "num_images": num_images,
            "grid_size": (rows, cols),
        }

    def temporal_comparison(
        self, images: List[np.ndarray], timestamps: List[str]
    ) -> Dict[str, Any]:
        """
        Compare images over time to detect changes.

        Args:
            images: List of images in chronological order
            timestamps: List of timestamps for each image

        Returns:
            Dictionary with temporal analysis
        """
        if len(images) < 2:
            return {"success": False, "error": "Need at least 2 images for comparison"}

        changes = []
        for i in range(len(images) - 1):
            diff_result = self.create_difference_map(images[i], images[i + 1])
            changes.append(
                {
                    "from_timestamp": timestamps[i],
                    "to_timestamp": timestamps[i + 1],
                    "similarity_score": diff_result["similarity_score"],
                    "difference_percentage": diff_result["difference_percentage"],
                }
            )

        # Calculate overall change trend
        avg_similarity = np.mean([c["similarity_score"] for c in changes])
        total_change = sum(c["difference_percentage"] for c in changes)

        return {
            "success": True,
            "num_images": len(images),
            "changes": changes,
            "average_similarity": float(avg_similarity),
            "total_change_percentage": float(total_change),
            "trend": "stable" if avg_similarity > 0.9 else "moderate" if avg_similarity > 0.7 else "significant",
        }

    def create_overlay_comparison(
        self, image1: np.ndarray, image2: np.ndarray, opacity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create overlay comparison with adjustable opacity.

        Args:
            image1: First image (BGR)
            image2: Second image (BGR)
            opacity: Opacity of second image (0.0-1.0)

        Returns:
            Dictionary with overlay image
        """
        # Ensure same size
        if image1.shape != image2.shape:
            h, w = image1.shape[:2]
            image2 = cv2.resize(image2, (w, h))

        # Create overlay
        overlay = cv2.addWeighted(image1, 1 - opacity, image2, opacity, 0)

        return {
            "success": True,
            "overlay": overlay,
            "opacity": opacity,
        }

    def create_slider_comparison(
        self, image1: np.ndarray, image2: np.ndarray, split_position: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create slider-style comparison (left/right split).

        Args:
            image1: First image (BGR)
            image2: Second image (BGR)
            split_position: Position of split (0.0-1.0)

        Returns:
            Dictionary with split comparison
        """
        # Ensure same size
        if image1.shape != image2.shape:
            h, w = image1.shape[:2]
            image2 = cv2.resize(image2, (w, h))

        h, w = image1.shape[:2]
        split_x = int(w * split_position)

        # Create split image
        split_image = image1.copy()
        split_image[:, split_x:] = image2[:, split_x:]

        # Draw split line
        cv2.line(split_image, (split_x, 0), (split_x, h), (255, 255, 0), 3)

        return {
            "success": True,
            "split_image": split_image,
            "split_position": split_position,
        }
