"""Image quality helpers for the webcam prototype."""
import cv2


def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def is_face_large_enough(face, minimum_dimension=120):
    x, y, width, height = face
    return min(width, height) >= minimum_dimension


def calculate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean()


def crop_face(frame, face):
    x, y, width, height = face
    return frame[y:y + height, x:x + width]


def resize_face(face_image, size=(112, 112)):
    return cv2.resize(face_image, size)
