from flask import Flask, request, render_template, url_for
from flask_cors import CORS
import os
from datetime import datetime
from class_pred import classify_image
from obj_count import universal_object_counter

app = Flask(__name__)
CORS(app)

# Upload folder inside static so images are browser-accessible
UPLOAD_FOLDER = os.path.join("backend", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
@app.route("/upload")#upload files here
def index():
    return render_template("upload.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file:
        return "No file uploaded", 400

    # Timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_filename = f"uploaded_{timestamp}.jpg"
    output_filename = f"object_detected_{timestamp}.jpg"

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

    # Save uploaded image
    file.save(input_path)

    # Classify
    object_class, confidence = classify_image(input_path)
    if confidence is None:
        confidence = 0.0

    # Count objects & save output image with rectangles
    count = universal_object_counter(input_path, output_path=output_path)

    # URL for displaying image in HTML
    image_url = url_for("static", filename=f"uploads/{output_filename}")

    return render_template(
        "result.html",
        object_class=object_class,
        count=count,
        confidence=confidence,
        image_url=image_url
    )

if __name__ == "__main__":
    app.run(debug=True)
