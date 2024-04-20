import torch
import cv2
import sys
from recognizer import FaceRecognizer
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

recognizer = FaceRecognizer(device=device)

if len(sys.argv) > 1 and sys.argv[1] == "register":
    print("Registering faces")
    recognizer.register()
    exit(0)


frame = cv2.imread("elon3.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(frame)

details = recognizer.recognize(img)

for name, min_dist, box in details:

    x1, y1, x2, y2 = box

    cv2.putText(frame,
                f"{name} {min_dist}", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# cv2.imshow("__", frame)

img = Image.fromarray(frame)
img.show()