"""
Revvel Forensic Studio — Metadata Extraction
Extract EXIF, IPTC, XMP, and other metadata from images and videos.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import os
from typing import Dict, Any, Optional
from datetime import datetime
import json


class MetadataExtractor:
    """Extract comprehensive metadata from media files."""

    def extract_all_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract all available metadata from file."""
        file_ext = os.path.splitext(file_path)[1].lower()

        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "file_type": file_ext,
            "extracted_at": datetime.utcnow().isoformat(),
        }

        if file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            result["exif"] = self.extract_exif(file_path)
            result["image_properties"] = self.extract_image_properties(file_path)
        elif file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
            result["video_metadata"] = self.extract_video_metadata(file_path)

        return result

    def extract_exif(self, image_path: str) -> Dict[str, Any]:
        """Extract EXIF metadata from image."""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()

            if not exif_data:
                return {"available": False, "reason": "No EXIF data found"}

            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)

                # Handle GPS data separately
                if tag == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id in value:
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag] = value[gps_tag_id]
                    exif_dict["GPSInfo"] = gps_data
                else:
                    # Convert bytes to string
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="ignore")
                    exif_dict[tag] = value

            # Extract key forensic fields
            forensic_fields = {
                "camera_make": exif_dict.get("Make", "Unknown"),
                "camera_model": exif_dict.get("Model", "Unknown"),
                "datetime_original": exif_dict.get("DateTimeOriginal", "Unknown"),
                "datetime_digitized": exif_dict.get("DateTimeDigitized", "Unknown"),
                "software": exif_dict.get("Software", "Unknown"),
                "orientation": exif_dict.get("Orientation", "Unknown"),
                "x_resolution": exif_dict.get("XResolution", "Unknown"),
                "y_resolution": exif_dict.get("YResolution", "Unknown"),
                "exposure_time": exif_dict.get("ExposureTime", "Unknown"),
                "f_number": exif_dict.get("FNumber", "Unknown"),
                "iso": exif_dict.get("ISOSpeedRatings", "Unknown"),
                "flash": exif_dict.get("Flash", "Unknown"),
                "focal_length": exif_dict.get("FocalLength", "Unknown"),
            }

            # Extract GPS coordinates if available
            gps_info = exif_dict.get("GPSInfo", {})
            if gps_info:
                forensic_fields["gps_coordinates"] = self._parse_gps_coordinates(gps_info)

            return {
                "available": True,
                "full_exif": exif_dict,
                "forensic_fields": forensic_fields,
            }

        except Exception as e:
            return {"available": False, "error": str(e)}

    def extract_image_properties(self, image_path: str) -> Dict[str, Any]:
        """Extract basic image properties."""
        try:
            image = Image.open(image_path)
            img_cv = cv2.imread(image_path)

            return {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "color_channels": img_cv.shape[2] if len(img_cv.shape) == 3 else 1,
                "bit_depth": image.bits if hasattr(image, "bits") else "Unknown",
                "file_size_bytes": os.path.getsize(image_path),
                "aspect_ratio": f"{image.width}:{image.height}",
            }
        except Exception as e:
            return {"error": str(e)}

    def extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract metadata from video file."""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {"available": False, "error": "Failed to open video"}

            metadata = {
                "available": True,
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
                "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                "file_size_bytes": os.path.getsize(video_path),
            }

            cap.release()
            return metadata

        except Exception as e:
            return {"available": False, "error": str(e)}

    def _parse_gps_coordinates(self, gps_info: Dict) -> Dict[str, Any]:
        """Parse GPS coordinates from EXIF GPS data."""
        try:
            lat = gps_info.get("GPSLatitude")
            lat_ref = gps_info.get("GPSLatitudeRef")
            lon = gps_info.get("GPSLongitude")
            lon_ref = gps_info.get("GPSLongitudeRef")

            if not all([lat, lat_ref, lon, lon_ref]):
                return {"available": False}

            # Convert to decimal degrees
            lat_decimal = self._convert_to_degrees(lat)
            if lat_ref == "S":
                lat_decimal = -lat_decimal

            lon_decimal = self._convert_to_degrees(lon)
            if lon_ref == "W":
                lon_decimal = -lon_decimal

            return {
                "available": True,
                "latitude": lat_decimal,
                "longitude": lon_decimal,
                "latitude_ref": lat_ref,
                "longitude_ref": lon_ref,
                "google_maps_url": f"https://www.google.com/maps?q={lat_decimal},{lon_decimal}",
            }

        except Exception as e:
            return {"available": False, "error": str(e)}

    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates to decimal degrees."""
        d, m, s = value
        return float(d) + float(m) / 60.0 + float(s) / 3600.0

    def detect_manipulation(self, image_path: str) -> Dict[str, Any]:
        """Detect potential image manipulation by analyzing metadata."""
        exif = self.extract_exif(image_path)

        if not exif.get("available"):
            return {
                "suspicious": True,
                "reason": "No EXIF data (possibly stripped or manipulated)",
                "confidence": "medium",
            }

        forensic_fields = exif.get("forensic_fields", {})
        suspicious_indicators = []

        # Check for missing critical fields
        if forensic_fields.get("datetime_original") == "Unknown":
            suspicious_indicators.append("Missing original datetime")

        if forensic_fields.get("camera_make") == "Unknown":
            suspicious_indicators.append("Missing camera make")

        # Check for software editing
        software = forensic_fields.get("software", "")
        if any(
            editor in software.lower()
            for editor in ["photoshop", "gimp", "paint", "editor"]
        ):
            suspicious_indicators.append(f"Edited with: {software}")

        # Check for inconsistent dates
        datetime_original = forensic_fields.get("datetime_original")
        datetime_digitized = forensic_fields.get("datetime_digitized")
        if (
            datetime_original != "Unknown"
            and datetime_digitized != "Unknown"
            and datetime_original != datetime_digitized
        ):
            suspicious_indicators.append("Inconsistent datetime fields")

        if len(suspicious_indicators) == 0:
            return {
                "suspicious": False,
                "confidence": "high",
                "indicators": [],
            }

        return {
            "suspicious": True,
            "confidence": "high" if len(suspicious_indicators) >= 3 else "medium",
            "indicators": suspicious_indicators,
        }

    def compare_metadata(
        self, file1_path: str, file2_path: str
    ) -> Dict[str, Any]:
        """Compare metadata between two files."""
        meta1 = self.extract_all_metadata(file1_path)
        meta2 = self.extract_all_metadata(file2_path)

        differences = []
        similarities = []

        # Compare EXIF data if both have it
        if meta1.get("exif", {}).get("available") and meta2.get("exif", {}).get("available"):
            forensic1 = meta1["exif"]["forensic_fields"]
            forensic2 = meta2["exif"]["forensic_fields"]

            for key in forensic1.keys():
                if forensic1[key] == forensic2.get(key):
                    similarities.append(f"{key}: {forensic1[key]}")
                else:
                    differences.append(
                        f"{key}: {forensic1[key]} vs {forensic2.get(key, 'N/A')}"
                    )

        return {
            "file1": file1_path,
            "file2": file2_path,
            "differences": differences,
            "similarities": similarities,
            "match_score": len(similarities) / (len(similarities) + len(differences))
            if (len(similarities) + len(differences)) > 0
            else 0,
        }

    def export_metadata_report(
        self, file_path: str, output_path: str, format: str = "json"
    ) -> bool:
        """Export metadata to file."""
        metadata = self.extract_all_metadata(file_path)

        try:
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
            elif format == "txt":
                with open(output_path, "w") as f:
                    f.write(f"Metadata Report for: {file_path}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(json.dumps(metadata, indent=2, default=str))
            else:
                return False

            return True

        except Exception:
            return False
