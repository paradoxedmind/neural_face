from facenet_pytorch import InceptionResnetV1, MTCNN
from types import MethodType
import torch
import os
import cv2 as cv
from PIL import Image

print("Loading Models")
mtcnn = MTCNN(
    image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
resnet = InceptionResnetV1(pretrained="vggface2").eval()


def encode(img):
    res = resnet(img)
    return res


def detect_box(self, img, save_path=None):
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)

    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes,
            batch_probs,
            batch_points,
            img,
            method=self.selection_method,
        )
    faces = self.extract(img, batch_boxes, save_path)

    return batch_boxes, faces


mtcnn.detect_box = MethodType(detect_box, mtcnn)

saved_pictures = "./saved"
all_people_faces = {}
files = os.listdir(saved_pictures)

for file in files:
    person_face, _ = file.split(".")
    img = cv.imread(f"{saved_pictures}/{person_face}.jpg")
    cropped = mtcnn(img)
    if cropped is not None:
        all_people_faces[person_face] = encode(cropped)[0, :]


def detect(cam=0, thres=0.7):
    vid = cv.VideoCapture(cam)
    while vid.grab():
        _, img = vid.retrieve()
        img = cv.flip(img, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        batch_boxes, cropped_images = mtcnn.detect_box(img)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = map(int, box)
                img_embedding = encode(cropped.unsqueeze(0))
                detect_dict = {}
                for k, v in all_people_faces.items():
                    # getting euclidean distances b/w face embeddings of detected faces and
                    # embeddings of known faces
                    detect_dict[k] = (v - img_embedding).norm().item()
                min_key = min(detect_dict, key=detect_dict.get)

                if detect_dict[min_key] >= thres:
                    min_key = "Undetected"

                cv.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
                cv.putText(
                    img,
                    min_key,
                    (x + 5, y + 10),
                    cv.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )

        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow("output", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            vid.release()
            cv.destroyAllWindows()
            break


detect(0)
