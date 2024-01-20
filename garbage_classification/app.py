import torch
from torchvision.io import read_image
from torchvision import transforms
from ultralytics import YOLO
from flask import Flask, request
from flask_cors import CORS
from PIL import Image

model_classify = torch.jit.load('garbage_jit.pt')
model_detect = 
print("Model loaded")

# Define class names, image size and transform function
cs = ['Organic', 'Renewable']
IMG_SIZE = (384,384)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE,antialias=True),
    transforms.ToTensor(),
])

app = Flask(__name__)
CORS(app)

@app.route("/api/classify/predict", methods=["POST"])
def predictImage():
    image = request.files["img"]
    if image is None:
        return {"message": "No image received"}
    im = Image.open(image)
    # The image can be converted to tensor using
    im = transform(im)
    im = im.to("cpu")
    pred = model_classify(im.unsqueeze(0))
    pred_cls = cs[pred.argmax(1)]

    return {"class": pred_cls}


@app.route("/api/detect/image", methods=["POST"])
def detectImage():
    image = request.files["img"]
    if image is None:
        return {"message": "No image received"}

    print("[*]Detecting Image")
    result = model_detect([Image.open(image)], max_det=1,conf=0.6, iou=0.8, nms=True)[0]
    print("[.]Detection complete")
    data = result.boxes.data.cpu().tolist()
    h, w = result.orig_shape

    names = result.names
    res = False
    if len(names)>0:
        res = True
        r = []
        for row in data:
            box = [row[0] / w, row[1] / h, row[2] / w, row[3] / h]
            conf = row[4]
            classId = int(row[5])
            name = names[classId]
            r.append(
                {
                    "box": box,
                    "confidence": conf,
                    "classId": classId,
                    "name": name,
                    "color": "#%02x%02x%02x",
                }
            )

        names = result.names
    return {"res":res,"frame": r}


@app.route("/api/detect/video", methods=["POST"])
def detectVideo():
    video = request.files["vid"]

    if video is None:
        return {"message": "No video received"}

    name = "/tmp/" + str(uuid.uuid4()) + ".mp4"
    video.save(name)

    predicted = model_detect(name, stream=True, conf=0.6, save=True)

    cap = cv2.VideoCapture(name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    uniqueClasses = {}
    totalDetectedDistinctClasses = []
    while True:
        frameResult = []
        try:
            result = next(predicted)

            data = result.boxes.data.cpu().tolist()
            h, w = result.orig_shape

            names = result.names

            for row in data:
                box = [row[0] / w, row[1] / h, row[2] / w, row[3] / h]
                conf = row[4]
                classId = int(row[5])
                name = names[classId]
                frameResult.append(
                    {
                        "box": box,
                        "confidence": conf,
                        "classId": classId,
                        "name": name,
                        "color": "#%02x%02x%02x",
                    }
                )
                uniqueClasses[classId] = name

            frames.append(frameResult)
        except StopIteration:
           break
    for k, v in uniqueClasses.items():
        totalDetectedDistinctClasses.append([k, v])
        print(
            "Total frames: ",
            len(frames),
            "\n Total Classes: ",
            len(totalDetectedDistinctClasses),
        )
        print(frames)
        return {
            "frames": frames,
            "classes": totalDetectedDistinctClasses,
            "fps": fps,
        }


@app.route("/")
def home():
    return {"message": "Moye moye"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
