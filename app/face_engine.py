"""InsightFace inference adapter. No biometric persistence."""
import os
from insightface.app import FaceAnalysis


def create_face_engine():
    model = os.getenv("SENTINEL_MODEL", "buffalo_s")
    engine = FaceAnalysis(
        name=model,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    engine.prepare(ctx_id=-1, det_size=(320, 320))
    return engine


def analyze_face(engine, frame):
    return engine.get(frame)
