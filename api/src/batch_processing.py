"""
Revvel Forensic Studio — Batch Processing
Process multiple images/videos in parallel for bulk evidence analysis.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import uuid

from .face_detection import FaceDetector
from .forensic_analysis import ForensicAnalyzer
from .beauty_enhancement import BeautyEnhancer


class BatchProcessor:
    """Process multiple files in parallel for forensic analysis."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.face_detector = FaceDetector()
        self.forensic_analyzer = ForensicAnalyzer()
        self.beauty_enhancer = BeautyEnhancer()

    def process_batch(
        self,
        file_paths: List[str],
        operations: List[str],
        output_dir: str,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple files in batch.

        Args:
            file_paths: List of file paths to process
            operations: List of operations to perform (e.g., ["face_detect", "forensic_analysis", "enhance"])
            output_dir: Directory to save results
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with batch results
        """
        batch_id = str(uuid.uuid4())
        results = {
            "batch_id": batch_id,
            "total_files": len(file_paths),
            "completed": 0,
            "failed": 0,
            "results": [],
            "started_at": datetime.utcnow().isoformat(),
        }

        os.makedirs(output_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, fp, operations, output_dir): fp
                for fp in file_paths
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results["results"].append(result)
                    results["completed"] += 1
                except Exception as e:
                    results["results"].append(
                        {
                            "file": file_path,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    results["failed"] += 1

                if progress_callback:
                    progress_callback(results["completed"], results["total_files"])

        results["completed_at"] = datetime.utcnow().isoformat()

        # Save batch report
        report_path = os.path.join(output_dir, f"batch_report_{batch_id}.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _process_single_file(
        self, file_path: str, operations: List[str], output_dir: str
    ) -> Dict[str, Any]:
        """Process a single file with specified operations."""
        file_name = os.path.basename(file_path)
        result = {
            "file": file_name,
            "status": "success",
            "operations": {},
        }

        try:
            # Face detection
            if "face_detect" in operations:
                faces = self.face_detector.detect_faces(file_path)
                result["operations"]["face_detect"] = {
                    "faces_found": len(faces),
                    "faces": faces,
                }

            # Forensic analysis
            if "forensic_analysis" in operations:
                analysis = self.forensic_analyzer.full_analysis(file_path)
                result["operations"]["forensic_analysis"] = analysis

            # Beauty enhancement
            if "enhance" in operations:
                enhanced_path = os.path.join(output_dir, f"enhanced_{file_name}")
                self.beauty_enhancer.enhance(file_path, enhanced_path)
                result["operations"]["enhance"] = {
                    "output_path": enhanced_path,
                }

            # Mask detection
            if "mask_detect" in operations:
                mask_result = self.forensic_analyzer.detect_mask(file_path)
                result["operations"]["mask_detect"] = mask_result

            # EXIF analysis
            if "exif" in operations:
                exif_data = self.forensic_analyzer.analyze_exif(file_path)
                result["operations"]["exif"] = exif_data

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def process_batch_async(
        self,
        file_paths: List[str],
        operations: List[str],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Async version of batch processing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.process_batch, file_paths, operations, output_dir, None
        )

    def get_batch_status(self, batch_id: str, output_dir: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch processing job."""
        report_path = os.path.join(output_dir, f"batch_report_{batch_id}.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                return json.load(f)
        return None

    def estimate_batch_time(self, num_files: int, operations: List[str]) -> float:
        """Estimate processing time in seconds."""
        # Base time per file
        time_per_file = 2.0

        # Add time for each operation
        operation_times = {
            "face_detect": 1.5,
            "forensic_analysis": 3.0,
            "enhance": 2.5,
            "mask_detect": 1.0,
            "exif": 0.5,
        }

        total_time_per_file = time_per_file + sum(
            operation_times.get(op, 1.0) for op in operations
        )

        # Account for parallel processing
        estimated_time = (num_files * total_time_per_file) / self.max_workers

        return estimated_time


class BatchQueue:
    """Manage a queue of batch processing jobs."""

    def __init__(self):
        self.queue: List[Dict[str, Any]] = []
        self.processing: Dict[str, Dict[str, Any]] = {}
        self.completed: Dict[str, Dict[str, Any]] = {}

    def add_job(
        self,
        file_paths: List[str],
        operations: List[str],
        output_dir: str,
        priority: int = 0,
    ) -> str:
        """Add a job to the queue."""
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "file_paths": file_paths,
            "operations": operations,
            "output_dir": output_dir,
            "priority": priority,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
        }
        self.queue.append(job)
        self.queue.sort(key=lambda x: x["priority"], reverse=True)
        return job_id

    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get the next job from the queue."""
        if self.queue:
            job = self.queue.pop(0)
            job["status"] = "processing"
            job["started_at"] = datetime.utcnow().isoformat()
            self.processing[job["job_id"]] = job
            return job
        return None

    def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark a job as completed."""
        if job_id in self.processing:
            job = self.processing.pop(job_id)
            job["status"] = "completed"
            job["completed_at"] = datetime.utcnow().isoformat()
            job["result"] = result
            self.completed[job_id] = job

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a job."""
        # Check queue
        for job in self.queue:
            if job["job_id"] == job_id:
                return job

        # Check processing
        if job_id in self.processing:
            return self.processing[job_id]

        # Check completed
        if job_id in self.completed:
            return self.completed[job_id]

        return None

    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "queued": len(self.queue),
            "processing": len(self.processing),
            "completed": len(self.completed),
        }
