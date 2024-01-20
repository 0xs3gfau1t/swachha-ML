from ultralytics import YOLO
from flask import Flask, request
from PIL import Image
import cv2
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


print("Loading model")
model = YOLO("./best.pt")
print("Model loaded")



@app.route("/api/image", methods=["POST"])
def predictImage():
    image = request.files["img"]
    if image is None:
        return {"message": "No image received"}

    print("[*]Detecting Image")
    result = model([Image.open(image)], max_det=1,conf=0.6, iou=0.8, nms=True)[0]
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


@app.route("/api/video", methods=["POST"])
def predictVideo():
    video = request.files["vid"]

    if video is None:
        return {"message": "No video received"}

    name = "/tmp/" + str(uuid.uuid4()) + ".mp4"
    video.save(name)

    predicted = model(name, stream=True, conf=0.6, save=True)

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
    app.run(host="0.0.0.0", port=3001, debug=True)
