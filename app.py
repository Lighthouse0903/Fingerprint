import os
import cv2
import numpy as np
import pyodbc
from flask import Flask, request, render_template, redirect, url_for
from skimage.morphology import skeletonize
from werkzeug.utils import secure_filename
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

from feature_extraction.segmentation import create_segmented_mask
from feature_extraction.orientation import calculate_orientation_field
from feature_extraction.frequency import calculate_frequency_field
from feature_extraction.enhancement import gabor_filter_enhancement
from feature_extraction.minutiae import extract_minutiae, remove_false_minutiae

UPLOAD_FOLDER = "static/query"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def minutiae_to_vector(minutiae):
    return np.array([[m["x"], m["y"], m["orientation"]] for m in minutiae])


def compare_minutiae(query_vec, db_vec):
    if len(db_vec) == 0 or len(query_vec) == 0:
        return 1e6
    distances = euclidean_distances(query_vec, db_vec)
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)


def extract_query_minutiae(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    roi_mask = create_segmented_mask(img, block_size=16, threshold_ratio=0.05)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_img = clahe.apply(img)
    norm_img = cv2.bitwise_and(norm_img, norm_img, mask=roi_mask)

    orientation_map = calculate_orientation_field(norm_img, roi_mask, block_size=16)
    frequency_map = calculate_frequency_field(
        norm_img, orientation_map, roi_mask, block_size=16
    )
    gabor_img = gabor_filter_enhancement(
        norm_img, orientation_map, frequency_map, roi_mask, block_size=16
    )
    gabor_img = cv2.bitwise_and(gabor_img, gabor_img, mask=roi_mask)

    binary = cv2.adaptiveThreshold(
        gabor_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask)
    skeleton = skeletonize(binary > 0)

    raw_minutiae = extract_minutiae(
        skeleton, roi_mask, orientation_map, 16, border_margin=15
    )
    return remove_false_minutiae(raw_minutiae, skeleton, roi_mask, orientation_map, 16)


def fetch_db_minutiae():
    conn = pyodbc.connect(
        r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-3G1MNP7\DWDM;DATABASE=FingerprintDB;UID=sa;PWD=123456"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, x, y, type, orientation FROM minutiae")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    image_minutiae = defaultdict(list)
    for row in rows:
        image_minutiae[row.image_name].append(
            {
                "x": row.x,
                "y": row.y,
                "type": row.type,
                "orientation": float(row.orientation),
            }
        )
    return image_minutiae


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file and file.filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        query_image_path = url_for("static", filename=f"query/{filename}")
        query_minutiae = extract_query_minutiae(filepath)
        query_vec = minutiae_to_vector(query_minutiae)
        db_minutiae = fetch_db_minutiae()

        results = []
        for img_name, minutiae in db_minutiae.items():
            db_vec = minutiae_to_vector(minutiae)
            score = compare_minutiae(query_vec, db_vec)
            results.append((img_name, score))

        results.sort(key=lambda x: x[1])
        top3 = results[:3]

        max_score = max(score for _, score in top3) + 1e-6
        top3_display = [
            {"image": name, "similarity": f"{(1 - score / max_score) * 100:.2f}"}
            for name, score in top3
        ]

        return render_template(
            "result.html",
            query_image=query_image_path,  # Gửi ảnh người dùng vào
            top3=top3_display,
        )

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
