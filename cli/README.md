# Revvel Forensic CLI

Standalone command-line tool for face detection, beauty enhancement, and forensic image analysis. Split from the `revvel-forensic-studio` monorepo and enhanced with Blue Ocean features.

## Features

### Core Capabilities
- **Face Detection** — Multi-backend (MediaPipe, dlib, OpenCV) face detection with landmark extraction
- **Beauty Enhancement** — YouCam Perfect-style presets (natural, glamour, dramatic, subtle)
- **Mask Detection** — Detect if a face is wearing a mask via contour/texture/edge analysis
- **Face Reconstruction** — Reconstruct faces from masked or obscured images
- **EXIF Analysis** — Extract and analyse metadata, detect manipulation indicators
- **Object Detection** — Glasses, jewelry, clothing tags, tattoos, text
- **Layer Decomposition** — Frequency-based layer extraction for forensic inspection
- **Edge Enhancement** — CLAHE + Canny edge strengthening
- **Zoom & Enhance** — Region-based upscaling with detail recovery
- **Face Swap** — Placeholder for face-swap operations

### Blue Ocean Enhancements
- **Batch Processing** — Process entire directories with optional parallel workers (`--workers`)
- **Video Frame Extraction** — Extract frames from video at configurable intervals
- **Report Generation** — Export forensic reports as **PDF**, **HTML**, or JSON
- **Batch Forensic Analysis** — Run full analysis across many images with a single command

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single image enhancement
python cli.py enhance -i photo.jpg -o enhanced.jpg --preset glamour

# Batch enhancement with 4 workers
python cli.py batch-enhance -i ./photos -o ./enhanced --workers 4

# Full forensic analysis with HTML report
python cli.py full-analysis -i evidence.jpg --report html

# Extract video frames
python cli.py extract-frames -i video.mp4 -o ./frames --interval 0.5

# Batch forensic analysis with PDF report
python cli.py batch-analyze -i ./evidence_dir -o ./reports --report pdf
```

## Docker

```bash
docker build -t revvel-forensic-cli .
docker run --rm -v $(pwd)/data:/data revvel-forensic-cli enhance -i /data/photo.jpg -o /data/out.jpg
```

## All Commands

| Command | Description |
|---|---|
| `enhance` | Apply beauty enhancements to a single image |
| `batch-enhance` | Batch-process a directory of images |
| `detect-mask` | Detect if a face is wearing a mask |
| `reconstruct-face` | Reconstruct face from masked image |
| `analyze-exif` | Analyse EXIF metadata |
| `detect-objects` | Detect glasses, jewelry, tags, tattoos |
| `decompose-layers` | Decompose image into frequency layers |
| `enhance-edges` | Enhance edges via lighting patterns |
| `zoom-enhance` | Zoom into a region and enhance |
| `face-swap` | Swap faces between two images |
| `full-analysis` | Full forensic analysis with report output |
| `extract-frames` | Extract frames from video (Blue Ocean) |
| `batch-analyze` | Batch forensic analysis (Blue Ocean) |

## Environment Variables

See `.env.example` for all supported configuration options.

## Architecture

```
revvel-forensic-cli/
├── cli.py                  # Main CLI entry point
├── src/
│   ├── __init__.py
│   ├── face_detection.py   # Face detection & mask detection
│   ├── beauty_enhancement.py  # Beauty/makeup engine
│   └── forensic_analysis.py   # Forensic tools
├── tests/
├── reports/                # Generated reports
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT License

## Credits

Developed by Revvel AI Engine — split from `revvel-forensic-studio` by Team 4.
