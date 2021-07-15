import pathlib
import base64
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from fastai.learner import load_learner
from fastai.vision.all import PILImage, Image, io, torch
from constants import ALLOWED_EXTENSIONS, SPECIES

app = Flask(__name__)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

@app.route('/', methods=["GET", "POST"])
def home():
    formatted_pred, errors, img_bytes = None, None, None

    if request.method == "POST":
        learn = load_learner('export.pkl', cpu=True)           
        f = request.files["file"]
        fname = secure_filename(f.filename)
        if validate_image_file(fname):
            # Call predict, return preds
            img = PILImage.create(f)
            img_bytes = img2io(f, fname)
            pred, _, probs = learn.predict(img)
            formatted_pred = f"{pred.replace('_', ' ')} ({round(torch.max(probs).item() * 100, 2)}%)"

        else:
            errors = "Unknown file type detected, please try again"    
    
    return render_template("home.html", list_birds=SPECIES, errors=errors, img=img_bytes, pred=formatted_pred)

def validate_image_file(fname):
    return "." in fname and fname.split(".")[1].lower() in ALLOWED_EXTENSIONS

def img2io(file, fname):
    """Converts image file to bytes to allow HTML rendering in-browser without saving the image"""
    ext = fname.split(".")[1].lower()
    secure_ext = "PNG" if ext == "png" else "JPEG"
    img, data = Image.open(file), io.BytesIO()
    img.save(data, secure_ext)
    encoded_img = base64.b64encode(data.getvalue())
    decoded_img = encoded_img.decode("utf-8")
    img_data = f"data:image/{secure_ext};base64, {decoded_img}"
    return img_data

if __name__ == '__main__':
    app.run()


