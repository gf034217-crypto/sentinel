"""Webcam face detection and on-demand embedding extraction."""
import cv2
from .quality import calculate_sharpness, calculate_brightness, is_face_large_enough, crop_face, resize_face
from .face_engine import create_face_engine, analyze_face

MIN_FACE_DIMENSION = 120
MIN_SHARPNESS = 100
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 200


def start_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Erro: não foi possível acessar a câmera.")
        return

    # Use the Haar cascade shipped with the installed OpenCV package.
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print("Erro: não foi possível carregar o detector facial.")
        camera.release()
        return

    engine = None
    recognition_requested = False
    print("SENTINEL iniciado. Pressione 'r' para analisar um rosto e 'q' para sair.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Erro ao capturar imagem.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
            sharpness = calculate_sharpness(frame)
            brightness = calculate_brightness(frame)
            sharpness_ok = sharpness >= MIN_SHARPNESS
            brightness_ok = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS

            cv2.putText(frame, f"Nitidez: {sharpness:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not len(faces):
                cv2.putText(frame, "SEM ROSTO DETECTADO", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Experimental demo: quality gate is applied to the largest face.
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                large_enough = is_face_large_enough((x, y, w, h), MIN_FACE_DIMENSION)
                if not large_enough:
                    status, color = "APROXIME-SE", (0, 255, 255)
                elif not sharpness_ok:
                    status, color = "IMAGEM DESFOCADA", (0, 0, 255)
                elif not brightness_ok:
                    status, color = "AJUSTE A ILUMINACAO", (0, 0, 255)
                else:
                    status, color = "QUALIDADE OK", (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, status, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if recognition_requested and large_enough and sharpness_ok and brightness_ok:
                    try:
                        if engine is None:
                            engine = create_face_engine()
                        result = analyze_face(engine, frame.copy())
                        print(f"InsightFace encontrou {len(result)} rosto(s).")
                        for face in result:
                            print("Bounding box:", face.bbox)
                            print("Landmarks:", face.kps)
                            print("Embedding shape:", face.embedding.shape)
                    except Exception as exc:
                        print(f"Falha na inferência: {exc}")
                    finally:
                        recognition_requested = False

            cv2.imshow("SENTINEL - Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                recognition_requested = True
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
