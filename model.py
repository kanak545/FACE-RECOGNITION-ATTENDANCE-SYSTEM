import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

# OpenCV Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------- Face Detection + Embedding --------
def crop_face_and_embed(bgr_image):

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (32, 32))

    emb = face.flatten().astype(np.float32) / 255.0

    return emb

# -------- Extract embedding from uploaded image --------
def extract_embedding_for_image(stream_or_bytes):

    data = stream_or_bytes.read()

    arr = np.frombuffer(data, np.uint8)

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    emb = crop_face_and_embed(img)

    return emb

# -------- Load model --------
def load_model_if_exists():

    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# -------- Predict --------
def predict_with_model(clf, emb):

    proba = clf.predict_proba([emb])[0]

    idx = np.argmax(proba)

    label = clf.classes_[idx]

    conf = float(proba[idx])

    return label, conf

# -------- Train Model --------
def train_model_background(dataset_dir, progress_callback=None):

    try:

        print("Training started")

        X = []
        y = []

        student_dirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]

        print("Students:", student_dirs)

        total_students = max(1, len(student_dirs))

        processed = 0

        for sid in student_dirs:

            folder = os.path.join(dataset_dir, sid)

            files = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for fn in files:

                path = os.path.join(folder, fn)

                img = cv2.imread(path)

                if img is None:
                    continue

                emb = crop_face_and_embed(img)

                if emb is None:
                    continue

                X.append(emb)

                y.append(int(sid))

            processed += 1

            if progress_callback:

                pct = int((processed / total_students) * 80)

                progress_callback(
                    pct,
                    f"Processed {processed}/{total_students} students"
                )

        if len(X) == 0:

            if progress_callback:
                progress_callback(0, "No training data found")

            return

        print("Embeddings:", len(X))

        X = np.stack(X)

        y = np.array(y)

        if progress_callback:
            progress_callback(85, "Training RandomForest...")

        print("Training RandomForest")

        clf = RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=42
        )

        clf.fit(X, y)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)

        print("Training complete")

        if progress_callback:
            progress_callback(100, "Training complete")

    except Exception as e:

        print("Training Error:", e)

        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")