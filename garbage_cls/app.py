import torch
from torchvision.io import read_image
from torchvision import transforms
from flask import Flask, request
from PIL import Image

model = torch.jit.load('garbage_jit.pt')
print("Model loaded")

# Define class names, image size and transform function
cs = ['Organic', 'Renewable']
IMG_SIZE = (384,384)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE,antialias=True),
    transforms.ToTensor(),
])

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def predictImage():
    image = request.files["img"]
    if image is None:
        return {"message": "No image received"}
    im = Image.open(image)
    # The image can be converted to tensor using
    im = transform(im)
    im = im.to("cpu")
    pred = model(im.unsqueeze(0))
    pred_cls = cs[pred.argmax(1)]

    return {"class": pred_cls}

@app.route("/")
def home():
    return {"message": "Moye moye"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
