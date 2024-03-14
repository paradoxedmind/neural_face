from facenet_pytorch import MTCNN
import cv2 as cv

mtcnn = MTCNN()

vid = cv.VideoCapture(0)
if vid is None:
    vid = cv.VideoCapture(1)
if vid is None:
    exit(0)

while vid.grab():
    _, img = vid.retrieve()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.flip(img, 1)

    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            box = map(int, box)
            x1, y1, x2, y2 = box
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 225, 0), 2)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow("window", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


vid.release()
cv.destroyAllWindows()
