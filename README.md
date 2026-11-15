# SENTINEL

**Status: experimental alpha — active development**

SENTINEL is a Python computer-vision prototype for webcam-based face detection, quality feedback, and on-demand facial embedding extraction using OpenCV and InsightFace. It is **not** a production-ready biometric authentication or access-control system.

## Current functionality

- Webcam capture and face detection using OpenCV's bundled Haar cascade
- Face-size, Laplacian sharpness, and brightness quality checks
- On-demand InsightFace inference (bounding box, landmarks and embedding shape)
- CPU inference with the `buffalo_s` model pack by default
- Standalone benchmarking with a locally supplied image

**Not implemented:** enrollment, a registered identity gallery, persistence, identity verification, liveness/anti-spoofing, calibrated thresholds, bias evaluation, or production deployment.

## Getting started

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

Use **r** to request facial analysis once quality checks pass; use **q** to quit. No identity is recognized or registered. An initial model download may occur; check upstream model licenses before any commercial use.

Benchmark using a local image that you have permission to process:

```bash
python benchmark.py path/to/image.jpg
```

## Project structure

- `app/main.py` — entry point
- `app/camera.py` — webcam loop and quality gate
- `app/quality.py` — image-quality helpers
- `app/face_engine.py` — InsightFace adapter
- `benchmark.py` — exploratory inference benchmark
- `SECURITY.md` — vulnerability reporting and biometric privacy guidance

## Privacy and security

No photographs, biometric embeddings, training data, credentials, model weights, or virtual environments are included. This code does not intentionally persist camera frames or embeddings. **Embeddings are sensitive biometric information** even without original photos. Obtain informed permission and comply with applicable data protection law before processing people's faces. Do not treat this prototype as an authentication system.

## Roadmap

1. Automated tests and reproducible benchmark metadata
2. Multi-sample enrollment with deletion of transient raw images
3. Identity gallery, cosine matching and calibrated decision thresholds
4. Anti-spoofing and threat-model evaluation
5. Privacy review, error handling and deployment hardening

## Technical limitations

Image-quality thresholds are heuristic and uncalibrated. Detection and recognition are separate components. Results depend on camera, CPU, model and input. No accuracy or security guarantees are made. Webcam execution and model loading require testing in the target environment.

## Licensing

No project-wide open-source license has been selected. The code is publicly visible, but reuse rights are not granted by default. External OpenCV and InsightFace components have their own license terms.
