from feature_extraction.segmentation import create_segmented_mask
from feature_extraction.orientation import calculate_orientation_field
from feature_extraction.frequency import calculate_frequency_field
from feature_extraction.enhancement import gabor_filter_enhancement
from feature_extraction.minutiae import extract_minutiae, remove_false_minutiae
import os
import cv2
import pyodbc
from pathlib import Path
from skimage.morphology import skeletonize

server = "DESKTOP-3G1MNP7\DWDM"
database = "FingerprintDB"
username = "sa"
password = "123456"

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={server};DATABASE={database};UID={username};PWD={password}"
)
cursor = conn.cursor()

cursor.execute(
    """
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'minutiae')
    CREATE TABLE minutiae (
        id INT IDENTITY(1,1) PRIMARY KEY,
        image_name NVARCHAR(255),
        x INT,
        y INT,
        type NVARCHAR(50),
        orientation FLOAT
    )
"""
)
conn.commit()

dataset_path = "./Dataset"
image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
image_paths = [
    os.path.join(dataset_path, f)
    for f in os.listdir(dataset_path)
    if Path(f).suffix.lower() in image_extensions
]

for img_path in image_paths:
    image_name = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    roi_mask = create_segmented_mask(img, block_size=16, threshold_ratio=0.05)
    img_roi = cv2.bitwise_and(img, img, mask=roi_mask)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_img_clahe = clahe.apply(img)
    norm_img_clahe_roi = cv2.bitwise_and(norm_img_clahe, norm_img_clahe, mask=roi_mask)
    orientation_map_block = calculate_orientation_field(
        norm_img_clahe_roi, roi_mask, block_size=16
    )
    frequency_map_block = calculate_frequency_field(
        norm_img_clahe_roi, orientation_map_block, roi_mask, block_size=16
    )
    gabor_enhanced_img = gabor_filter_enhancement(
        norm_img_clahe_roi,
        orientation_map_block,
        frequency_map_block,
        roi_mask,
        block_size=16,
    )
    gabor_enhanced_img = cv2.bitwise_and(
        gabor_enhanced_img, gabor_enhanced_img, mask=roi_mask
    )
    binary_img = cv2.adaptiveThreshold(
        gabor_enhanced_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    binary_img = cv2.bitwise_and(binary_img, binary_img, mask=roi_mask)
    thinned_skeleton = skeletonize(binary_img > 0)

    raw_minutiae = extract_minutiae(
        thinned_skeleton, roi_mask, orientation_map_block, 16, border_margin=15
    )
    cleaned_minutiae = remove_false_minutiae(
        raw_minutiae,
        thinned_skeleton,
        roi_mask,
        orientation_map_block,
        16,
        min_distance_between_minutiae=8,
        short_ridge_max_length=12,
    )

    for m in cleaned_minutiae:
        cursor.execute(
            """
            INSERT INTO minutiae (image_name, x, y, type, orientation)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                str(image_name),
                int(m["x"]),
                int(m["y"]),
                str(m["type"]),
                float(m["orientation"]),  # Ép kiểu float Python chuẩn
            ),
        )

    conn.commit()

print("✅ Đã lưu toàn bộ đặc trưng minutiae vào SQL Server.")

cursor.close()
conn.close()
