#!/usr/bin/env python3
"""
Revvel Forensic CLI — Standalone command-line tool for face detection,
enhancement, and forensic analysis.

Blue Ocean enhancements:
  - Batch processing with parallel workers
  - Video frame extraction
  - Report generation (PDF / HTML)
"""

import sys
import os
import json
import time
import datetime
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
import cv2
import numpy as np
from tqdm import tqdm

from src.face_detection import FaceDetector, MaskDetector, load_image, save_image
from src.beauty_enhancement import BeautyEnhancer, BatchProcessor
from src.forensic_analysis import (
    ForensicAnalyzer, FaceReconstructor, EXIFAnalyzer,
    ObjectDetector, LayerAnalyzer, EdgeEnhancer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _collect_images(directory: str, extensions: str = "jpg,jpeg,png,bmp,tiff") -> List[Path]:
    """Recursively collect image files from *directory*."""
    exts = [e.strip().lower() for e in extensions.split(",")]
    results: List[Path] = []
    for ext in exts:
        results.extend(Path(directory).rglob(f"*.{ext}"))
        results.extend(Path(directory).rglob(f"*.{ext.upper()}"))
    return sorted(set(results))


# ---------------------------------------------------------------------------
# Report generator (Blue Ocean)
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate PDF and HTML forensic reports."""

    @staticmethod
    def generate_html(data: Dict[str, Any], title: str = "Forensic Report",
                      output_path: str = "report.html") -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sections = ""
        for key, value in data.items():
            sections += f"<h2>{key.replace('_', ' ').title()}</h2>\n"
            sections += f"<pre>{json.dumps(value, indent=2, default=str)}</pre>\n"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  pre {{ background: #fff; padding: 16px; border-radius: 8px; overflow-x: auto;
         border: 1px solid #ddd; font-size: 13px; line-height: 1.5; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  .footer {{ margin-top: 40px; padding-top: 10px; border-top: 1px solid #ccc;
             color: #999; font-size: 0.8em; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Generated: {ts} | Revvel Forensic CLI v2.0.0</p>
  {sections}
  <div class="footer">Revvel Forensic CLI &mdash; automated forensic report</div>
</body>
</html>"""
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

    @staticmethod
    def generate_pdf(data: Dict[str, Any], title: str = "Forensic Report",
                     output_path: str = "report.pdf") -> str:
        """Generate a PDF report via FPDF2."""
        try:
            from fpdf import FPDF
        except ImportError:
            click.echo("fpdf2 not installed – falling back to HTML report", err=True)
            return ReportGenerator.generate_html(data, title, output_path.replace(".pdf", ".html"))

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, title, ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", ln=True, align="C")
        pdf.ln(8)

        for key, value in data.items():
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, key.replace("_", " ").title(), ln=True)
            pdf.set_font("Courier", "", 9)
            text = json.dumps(value, indent=2, default=str)
            for line in text.split("\n"):
                pdf.cell(0, 5, line[:120], ln=True)
            pdf.ln(4)

        pdf.output(output_path)
        return output_path


# ---------------------------------------------------------------------------
# Video frame extractor (Blue Ocean)
# ---------------------------------------------------------------------------

class VideoFrameExtractor:
    """Extract frames from video files for forensic analysis."""

    @staticmethod
    def extract_frames(video_path: str, output_dir: str,
                       interval: float = 1.0, max_frames: Optional[int] = None) -> List[str]:
        """
        Extract frames at *interval* seconds apart.

        Args:
            video_path: Path to video file.
            output_dir: Directory to save extracted frames.
            interval: Seconds between extracted frames.
            max_frames: Cap on total frames extracted.

        Returns:
            List of saved frame paths.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(fps * interval))

        os.makedirs(output_dir, exist_ok=True)
        saved: List[str] = []
        idx = 0

        with tqdm(total=min(total_frames // frame_step, max_frames or float("inf")),
                  desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_step == 0:
                    out_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
                    cv2.imwrite(out_path, frame)
                    saved.append(out_path)
                    pbar.update(1)
                    if max_frames and len(saved) >= max_frames:
                        break
                idx += 1

        cap.release()
        return saved


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="2.0.0")
def cli():
    """Revvel Forensic CLI — Image Processing & Forensic Analysis"""
    pass


# ===== Beauty Enhancement =====

@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, type=click.Path())
@click.option("--preset", "-p", default="natural",
              type=click.Choice(["natural", "glamour", "dramatic", "subtle"]))
@click.option("--skin-smooth", type=float, help="Skin smoothing 0-1")
@click.option("--brightness", type=float, help="Brightness 0.8-1.2")
def enhance(input_path, output_path, preset, skin_smooth, brightness):
    """Apply beauty enhancements to an image."""
    image = cv2.imread(input_path)
    if image is None:
        click.echo(f"Error: cannot load {input_path}", err=True)
        sys.exit(1)

    enhancer = BeautyEnhancer()
    custom = {}
    if skin_smooth is not None:
        custom["skin_smooth"] = skin_smooth
    if brightness is not None:
        custom["brightness"] = brightness

    enhanced = enhancer.enhance(image, preset=preset,
                                custom_params=custom if custom else None)
    cv2.imwrite(output_path, enhanced)
    click.echo(f"Enhanced image saved: {output_path}")


@cli.command("batch-enhance")
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
@click.option("--preset", "-p", default="natural",
              type=click.Choice(["natural", "glamour", "dramatic", "subtle"]))
@click.option("--count", "-c", type=int, help="Max images to process")
@click.option("--workers", "-w", type=int, default=1, help="Parallel workers (Blue Ocean)")
@click.option("--extensions", "-e", default="jpg,jpeg,png")
def batch_enhance(input_dir, output_dir, preset, count, workers, extensions):
    """Batch-process images with optional parallelism."""
    images = _collect_images(input_dir, extensions)
    if count:
        images = images[:count]
    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Found {len(images)} images — workers={workers}")

    def _process_one(img_path: Path) -> str:
        img = cv2.imread(str(img_path))
        if img is None:
            return ""
        enh = BeautyEnhancer()
        result = enh.enhance(img, preset=preset)
        out = os.path.join(output_dir, f"enhanced_{img_path.name}")
        cv2.imwrite(out, result)
        return out

    done = 0
    with tqdm(total=len(images), desc="Batch enhance") as pbar:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process_one, p): p for p in images}
                for f in as_completed(futures):
                    pbar.update(1)
                    if f.result():
                        done += 1
        else:
            for p in images:
                if _process_one(p):
                    done += 1
                pbar.update(1)

    click.echo(f"Batch complete — {done}/{len(images)} processed")


# ===== Forensic Analysis =====

@cli.command("detect-mask")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", type=click.Path())
def detect_mask(input_path, output_path):
    """Detect if a face is wearing a mask."""
    image = cv2.imread(input_path)
    if image is None:
        click.echo(f"Error: cannot load {input_path}", err=True)
        sys.exit(1)

    detector = MaskDetector()
    result = detector.detect_mask(image)

    click.echo("\n=== Mask Detection ===")
    click.echo(f"Detected: {result['mask_detected']}")
    click.echo(f"Confidence: {result['confidence']:.2%}")
    if "reasons" in result:
        for r in result["reasons"]:
            click.echo(f"  - {r}")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2, default=str))
        click.echo(f"Results saved: {output_path}")


@cli.command("reconstruct-face")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, type=click.Path())
def reconstruct_face(input_path, output_path):
    """Reconstruct face from masked/obscured image."""
    image = cv2.imread(input_path)
    if image is None:
        click.echo(f"Error: cannot load {input_path}", err=True)
        sys.exit(1)

    recon = FaceReconstructor()
    result = recon.reconstruct_from_masked(image)

    if result["success"]:
        cv2.imwrite(output_path, result["reconstructed"])
        click.echo(f"Reconstructed face saved: {output_path} (confidence {result['confidence']:.2%})")
    else:
        click.echo(f"Reconstruction failed: {result['reason']}", err=True)
        sys.exit(1)


@cli.command("analyze-exif")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", type=click.Path())
def analyze_exif(input_path, output_path):
    """Analyze EXIF metadata."""
    analyzer = EXIFAnalyzer()
    result = analyzer.analyze(input_path)

    click.echo("\n=== EXIF Analysis ===")
    for section in ("camera", "location", "datetime", "software"):
        val = result.get(section)
        if val:
            click.echo(f"{section.title()}: {val}")
    if result.get("manipulation_indicators"):
        click.echo("Manipulation indicators:")
        for ind in result["manipulation_indicators"]:
            click.echo(f"  ! {ind}")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2, default=str))
        click.echo(f"Saved: {output_path}")


@cli.command("detect-objects")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", type=click.Path())
def detect_objects(input_path, output_path):
    """Detect objects (glasses, jewelry, tags, tattoos)."""
    image = cv2.imread(input_path)
    if image is None:
        click.echo(f"Error: cannot load {input_path}", err=True)
        sys.exit(1)

    detector = ObjectDetector()
    result = detector.detect_all(image)

    click.echo("\n=== Object Detection ===")
    for obj_type, data in result.items():
        if data["detected"]:
            click.echo(f"  {obj_type}: count={data['count']}")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2, default=str))
        click.echo(f"Saved: {output_path}")


@cli.command("decompose-layers")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
@click.option("--num-layers", "-n", default=5, type=int)
def decompose_layers(input_path, output_dir, num_layers):
    """Decompose image into frequency layers."""
    image = cv2.imread(input_path)
    if image is None:
        click.echo(f"Error: cannot load {input_path}", err=True)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    analyzer = LayerAnalyzer()
    layers = analyzer.decompose(image, num_layers=num_layers)

    for i, layer in enumerate(layers):
        p = os.path.join(output_dir, f"layer_{i:02d}.png")
        cv2.imwrite(p, layer)
        click.echo(f"  Layer {i}: {p}")

    click.echo(f"All {len(layers)} layers saved to {output_dir}")


@cli.command("enhance-edges")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, type=click.Path())
@click.option("--strength", "-s", default=1.5, type=float)
def enhance_edges(input_path, output_path, strength):
    """Enhance edges using lighting and makeup patterns."""
    image = cv2.imread(input_path)
    if image is None:
        click.echo(f"Error: cannot load {input_path}", err=True)
        sys.exit(1)

    enhancer = EdgeEnhancer()
    enhanced = enhancer.enhance_facial_edges(image, strength=strength)
    cv2.imwrite(output_path, enhanced)
    click.echo(f"Edge-enhanced image saved: {output_path}")


@cli.command("zoom-enhance")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, type=click.Path())
@click.option("--x", type=int, required=True)
@click.option("--y", type=int, required=True)
@click.option("--width", type=int, required=True)
@click.option("--height", type=int, required=True)
@click.option("--scale", "-s", default=2.0, type=float)
def zoom_enhance(input_path, output_path, x, y, width, height, scale):
    """Zoom into a region and enhance details."""
    image = cv2.imread(input_path)
    if image is None:
        click.echo(f"Error: cannot load {input_path}", err=True)
        sys.exit(1)

    enhancer = EdgeEnhancer()
    enhanced = enhancer.zoom_and_enhance(image, (x, y, width, height), scale=scale)
    cv2.imwrite(output_path, enhanced)
    click.echo(f"Zoomed region saved: {output_path}")


@cli.command("face-swap")
@click.option("--source", "-s", required=True, type=click.Path(exists=True))
@click.option("--target", "-t", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, type=click.Path())
def face_swap(source, target, output_path):
    """Swap faces between two images."""
    src = cv2.imread(source)
    tgt = cv2.imread(target)
    if src is None or tgt is None:
        click.echo("Error: cannot load images", err=True)
        sys.exit(1)

    click.echo("Face swap (placeholder — requires dlib shape predictor for full impl)")
    cv2.imwrite(output_path, tgt)
    click.echo(f"Output saved: {output_path}")


@cli.command("full-analysis")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", type=click.Path())
@click.option("--report", "-r", type=click.Choice(["json", "html", "pdf"]), default="json",
              help="Report format (Blue Ocean)")
def full_analysis(input_path, output_path, report):
    """Perform full forensic analysis with optional report generation."""
    click.echo(f"Running full forensic analysis on {input_path} ...")
    analyzer = ForensicAnalyzer()
    result = analyzer.full_analysis(input_path)

    # Serialise numpy
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    result = _convert(result)

    click.echo("\n=== Full Forensic Analysis ===")
    if result.get("exif", {}).get("camera"):
        click.echo(f"Camera: {result['exif']['camera']}")
    objs = [k for k, v in result.get("objects", {}).items() if v.get("detected")]
    if objs:
        click.echo(f"Objects: {', '.join(objs)}")
    if result.get("reconstruction"):
        click.echo(f"Reconstruction: {result['reconstruction'].get('recommendation', 'N/A')}")

    out = output_path or f"forensic_report_{_timestamp()}.{report}"
    if report == "html":
        ReportGenerator.generate_html(result, title=f"Forensic Report — {Path(input_path).name}", output_path=out)
    elif report == "pdf":
        ReportGenerator.generate_pdf(result, title=f"Forensic Report — {Path(input_path).name}", output_path=out)
    else:
        Path(out).write_text(json.dumps(result, indent=2, default=str))

    click.echo(f"Report saved: {out}")


# ===== Blue Ocean: Video frame extraction =====

@cli.command("extract-frames")
@click.option("--input", "-i", "video_path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
@click.option("--interval", default=1.0, type=float, help="Seconds between frames")
@click.option("--max-frames", type=int, help="Maximum frames to extract")
def extract_frames(video_path, output_dir, interval, max_frames):
    """Extract frames from a video file for analysis."""
    extractor = VideoFrameExtractor()
    frames = extractor.extract_frames(video_path, output_dir, interval=interval, max_frames=max_frames)
    click.echo(f"Extracted {len(frames)} frames to {output_dir}")


# ===== Blue Ocean: Batch forensic analysis =====

@cli.command("batch-analyze")
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
@click.option("--report", "-r", type=click.Choice(["json", "html", "pdf"]), default="json")
@click.option("--count", "-c", type=int)
def batch_analyze(input_dir, output_dir, report, count):
    """Batch forensic analysis across a directory of images."""
    images = _collect_images(input_dir)
    if count:
        images = images[:count]
    os.makedirs(output_dir, exist_ok=True)

    analyzer = ForensicAnalyzer()
    summary: Dict[str, Any] = {"total": len(images), "results": []}

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    with tqdm(total=len(images), desc="Analyzing") as pbar:
        for img_path in images:
            try:
                result = analyzer.full_analysis(str(img_path))
                result = _convert(result)
                result["source_file"] = str(img_path)
                summary["results"].append(result)
            except Exception as e:
                summary["results"].append({"source_file": str(img_path), "error": str(e)})
            pbar.update(1)

    out = os.path.join(output_dir, f"batch_report_{_timestamp()}.{report}")
    if report == "html":
        ReportGenerator.generate_html(summary, title="Batch Forensic Report", output_path=out)
    elif report == "pdf":
        ReportGenerator.generate_pdf(summary, title="Batch Forensic Report", output_path=out)
    else:
        Path(out).write_text(json.dumps(summary, indent=2, default=str))

    click.echo(f"Batch analysis complete — report: {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli()
