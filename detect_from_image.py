from facenet_pytorch import MTCNN
from PIL import Image,ImageDraw

# Intialize MTCNN for face detection
mtcnn = MTCNN()

# Load an image containing faces
image = Image.open("./1.jpg")

# Detect faces in the image
boxes, _ = mtcnn.detect(image)

if boxes is not None:
    for box in boxes:
        # Draw bounding boxe on image
        draw = ImageDraw.Draw(image)
        draw.rectangle(box.tolist(), outline='red', width=3)

image.show()
