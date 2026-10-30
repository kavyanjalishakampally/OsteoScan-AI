
import os
import sqlite3
import uuid
import hmac
from datetime import datetime
from io import BytesIO
from functools import wraps

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_file, abort, session
)
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img, img_to_array
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "model", "osteoporosis_efficientnet_best_v2.keras"
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "osteoscan.db")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["SECRET_KEY"] = os.environ.get(
    "OSTEOSCAN_SECRET_KEY",
    "local-development-key-change-before-deployment"
)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

LOGIN_USERNAME = os.environ.get("OSTEOSCAN_USERNAME", "admin")
LOGIN_PASSWORD = os.environ.get("OSTEOSCAN_PASSWORD", "OsteoScan@123")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Login protection
def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)
    return wrapped_view


# Database
def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    new_columns = {
        "xray_type": "TEXT DEFAULT 'single'",
        "left_prediction": "TEXT",
        "left_confidence": "REAL",
        "right_prediction": "TEXT",
        "right_confidence": "REAL",
        "left_image_file": "TEXT",
        "right_image_file": "TEXT",
    }

    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_age TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_file TEXT,
                created_at TEXT NOT NULL
            )
        """)

        existing = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }

        for column, definition in new_columns.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE predictions ADD COLUMN "
                    f"{column} {definition}"
                )

    print("Prediction database ready!")


init_database()

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def classify_image(image_path):
    """Return the model's class and its class score percentage."""
    img = load_img(
        image_path, target_size=(224, 224), color_mode="rgb"
    )
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    probability = float(model.predict(img_array, verbose=0)[0][0])

    if probability >= 0.5:
        return "Osteoporosis", probability * 100

    return "Normal", (1 - probability) * 100


def make_knee_crops(image_path, left_path, right_path):
    """
    Experimental fixed crops for bilateral images.
    These coordinates must be verified for the X-ray layout.
    """
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size

        if width < 100 or height < 100:
            raise ValueError("The uploaded X-ray image is too small.")

        left_box = (
            int(width * 0.10),
            int(height * 0.15),
            int(width * 0.48),
            int(height * 0.80),
        )
        right_box = (
            int(width * 0.52),
            int(height * 0.15),
            int(width * 0.90),
            int(height * 0.80),
        )

        image.crop(left_box).save(left_path)
        image.crop(right_box).save(right_path)


# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        username_ok = hmac.compare_digest(
            username.encode("utf-8"),
            LOGIN_USERNAME.encode("utf-8")
        )
        password_ok = hmac.compare_digest(
            password.encode("utf-8"),
            LOGIN_PASSWORD.encode("utf-8")
        )

        if username_ok and password_ok:
            session.clear()
            session["logged_in"] = True
            session["username"] = LOGIN_USERNAME
            return redirect(url_for("home"))

        error = "Invalid username or password. Please try again."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    patient_id = request.form.get("patient_id", "").strip()
    patient_age = request.form.get("patient_age", "").strip()
    xray_type = request.form.get("xray_type", "single").strip()

    if not patient_id:
        return "Please enter a patient reference ID.", 400

    if len(patient_id) > 40:
        return "Patient reference ID must be 40 characters or fewer.", 400

    if patient_age and (
        not patient_age.isdigit()
        or not 1 <= int(patient_age) <= 120
    ):
        return "Please enter a valid age between 1 and 120.", 400

    if xray_type not in {"single", "bilateral"}:
        return "Invalid X-ray type selected.", 400

    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        return "Please select an X-ray image.", 400

    if not allowed_file(uploaded.filename):
        return "Please upload a JPG, JPEG, or PNG image.", 400

    safe_name = secure_filename(uploaded.filename)
    if not safe_name:
        return "Invalid filename.", 400

    extension = safe_name.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    left_path = None
    right_path = None

    try:
        uploaded.save(file_path)

        left_prediction = None
        left_confidence = None
        right_prediction = None
        right_confidence = None
        left_image_file = None
        right_image_file = None

        if xray_type == "single":
            prediction, confidence = classify_image(file_path)

        else:
            left_filename = f"{uuid.uuid4().hex}_left.png"
            right_filename = f"{uuid.uuid4().hex}_right.png"

            left_path = os.path.join(UPLOAD_FOLDER, left_filename)
            right_path = os.path.join(UPLOAD_FOLDER, right_filename)

            make_knee_crops(file_path, left_path, right_path)

            left_prediction, left_confidence = classify_image(left_path)
            right_prediction, right_confidence = classify_image(right_path)

            # Summary fields for compatibility with existing history.
            prediction = (
                f"Image-left: {left_prediction}; "
                f"Image-right: {right_prediction}"
            )
            confidence = (
                left_confidence + right_confidence
            ) / 2

            left_image_file = f"uploads/{left_filename}"
            right_image_file = f"uploads/{right_filename}"

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_db() as connection:
            cursor = connection.execute("""
                INSERT INTO predictions (
                    patient_id, patient_age, prediction, confidence,
                    image_file, created_at, xray_type,
                    left_prediction, left_confidence,
                    right_prediction, right_confidence,
                    left_image_file, right_image_file
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                patient_age or None,
                prediction,
                confidence,
                f"uploads/{filename}",
                created_at,
                xray_type,
                left_prediction,
                left_confidence,
                right_prediction,
                right_confidence,
                left_image_file,
                right_image_file,
            ))

            record_id = cursor.lastrowid

        return redirect(url_for("result", record_id=record_id))

    except Exception:
        app.logger.exception("Image prediction failed")
        return (
            "Unable to process the image. Check the terminal for details.",
            500
        )


# Result page
@app.route("/result/<int:record_id>")
@login_required
def result(record_id):
    with get_db() as connection:
        record = connection.execute(
            "SELECT * FROM predictions WHERE id = ?", (record_id,)
        ).fetchone()

    if record is None:
        abort(404)

    return render_template(
        "result.html",
        prediction=record["prediction"],
        confidence=record["confidence"],
        image_file=record["image_file"],
        patient_id=record["patient_id"],
        patient_age=record["patient_age"],
        record_id=record["id"],
        created_at=record["created_at"],
        xray_type=record["xray_type"],
        left_prediction=record["left_prediction"],
        left_confidence=record["left_confidence"],
        right_prediction=record["right_prediction"],
        right_confidence=record["right_confidence"],
        left_image_file=record["left_image_file"],
        right_image_file=record["right_image_file"],
    )


# History
@app.route("/history")
@login_required
def history():
    with get_db() as connection:
        records = connection.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()

    return render_template("history.html", records=records)


# PDF report
@app.route("/report/<int:record_id>")
@login_required
def download_report(record_id):
    with get_db() as connection:
        record = connection.execute(
            "SELECT * FROM predictions WHERE id = ?", (record_id,)
        ).fetchone()

    if record is None:
        abort(404)

    pdf_buffer = BytesIO()
    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph("OsteoScan AI", styles["Title"]),
        Paragraph("Knee X-ray Screening Report", styles["Heading2"]),
        Spacer(1, 18),
        Paragraph(
            "This report contains experimental AI screening results "
            "for educational and research use. It is not a diagnosis.",
            styles["BodyText"]
        ),
        Spacer(1, 18),
    ]

    rows = [
        ["Report field", "Details"],
        ["Report number", str(record["id"])],
        ["Patient reference ID", record["patient_id"]],
        ["Age", record["patient_age"] or "Not provided"],
        ["X-ray type", record["xray_type"] or "single"],
    ]

    if record["xray_type"] == "bilateral":
        rows.extend([
            ["Image-left knee result", record["left_prediction"] or "Unavailable"],
            ["Image-left model score", (
                f'{record["left_confidence"]:.2f}%'
                if record["left_confidence"] is not None else "Unavailable"
            )],
            ["Image-right knee result", record["right_prediction"] or "Unavailable"],
            ["Image-right model score", (
                f'{record["right_confidence"]:.2f}%'
                if record["right_confidence"] is not None else "Unavailable"
            )],
        ])
    else:
        rows.extend([
            ["Prediction", record["prediction"]],
            ["Model score", f'{record["confidence"]:.2f}%'],
        ])

    rows.append(["Date and time", record["created_at"]])

    table = Table(rows, colWidths=[2.0 * inch, 4.4 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5dce5")),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.extend([
        table,
        Spacer(1, 20),
        Paragraph(
            "Important: This experimental model can make errors. "
            "The percentages are model scores, not calibrated probabilities "
            "of disease. A qualified healthcare professional must interpret "
            "medical images and determine diagnosis or treatment.",
            styles["BodyText"]
        )
    ])

    document.build(story)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"osteoscan_report_{record_id}.pdf"
    )


if __name__ == "__main__":
    app.run(debug=True)
