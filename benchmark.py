"""Local-image inference benchmark. No sample images are distributed."""
import argparse
import statistics
import time
import cv2
from insightface.app import FaceAnalysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a local image used with permission")
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    image = cv2.imread(args.image)
    if image is None:
        parser.error("Could not open the specified image")

    engine = FaceAnalysis(name="buffalo_s", allowed_modules=["detection", "recognition"],
                          providers=["CPUExecutionProvider"])
    engine.prepare(ctx_id=-1, det_size=(320, 320))
    start = time.perf_counter()
    engine.get(image)
    first = time.perf_counter() - start
    times = []
    for index in range(args.repetitions):
        start = time.perf_counter()
        faces = engine.get(image)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {index + 1}: {elapsed:.3f}s | faces={len(faces)}")
    print(f"First inference: {first:.3f}s")
    print(f"Mean: {statistics.mean(times):.3f}s")
    print(f"Median: {statistics.median(times):.3f}s")
    print(f"Worst: {max(times):.3f}s")


if __name__ == "__main__":
    main()
