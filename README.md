# Revvel Forensic Studio

Comprehensive image processing and forensic analysis application combining YouCam Perfect-style beauty/enhancement features with advanced forensic investigation tools.

## Features

### Beauty & Enhancement (YouCam Perfect Clone)
- Face detection and beautification
- Makeup application (contour, highlight, lipstick, eyeshadow)
- Skin smoothing and blemish removal
- Face reshaping (chin, cheeks, nose, jawline)
- Body reshaping
- Teeth whitening
- Eye enhancement
- Hair color change
- Batch processing (20, 50, 100+ images)

### Forensic Analysis Tools
- **Mask Detection**: Analyze facial features to detect if someone is wearing a mask based on contour, highlight, and shape distortions around nose/mouth
- **Face Reconstruction**: Build facial features from partial/masked images using edge detection, lighting analysis, and makeup pattern recognition
- **Face Swap & Retake**: Advanced face swapping with noise-based reconstruction from masked/obscured faces
- **EXIF Analysis**: Extract, analyze, and recover EXIF metadata from images
- **Object Detection**: Detect glasses, jewelry, clothing tags, tattoos, and other identifying features
- **Zoom & Enhance**: Intelligent upscaling and detail enhancement for arms, wrists, sleeves, background objects
- **Layer Analysis**: Decompose images into layers and progressively remove thin layers to reveal underlying details
- **Edge Enhancement**: Enhance edges using lighting and makeup patterns to reconstruct obscured facial features
- **Forensic Comparison**: Side-by-side comparison tools for investigative analysis
- **Stingray/IMSI Analysis**: Integration with telecommunications forensics tools

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### CLI
```bash
# Beauty enhancement
python cli/main.py enhance --input photo.jpg --output enhanced.jpg --preset natural

# Batch processing
python cli/main.py batch-enhance --input-dir ./photos --output-dir ./enhanced --count 50

# Forensic analysis
python cli/main.py detect-mask --input suspect.jpg
python cli/main.py reconstruct-face --input masked_face.jpg --output reconstructed.jpg
python cli/main.py analyze-exif --input photo.jpg

# Face swap
python cli/main.py face-swap --source face1.jpg --target face2.jpg --output swapped.jpg
```

### API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### MCP Server
```bash
python mcp/server.py
```

## Architecture

- `src/` - Core modules (face detection, enhancement, forensics)
- `cli/` - Command-line interface
- `api/` - FastAPI REST API
- `mcp/` - Model Context Protocol server
- `tests/` - Unit and integration tests
- `docs/` - Documentation
- `examples/` - Example images and scripts

## Technologies

- OpenCV for image processing
- dlib for face detection and landmarks
- MediaPipe for advanced face mesh
- PIL/Pillow for image manipulation
- scikit-image for forensic analysis
- FastAPI for REST API
- PyInstaller for .exe packaging

## License

MIT License
