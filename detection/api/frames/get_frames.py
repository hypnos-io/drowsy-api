import cv2

# Open the video file
cap = cv2.VideoCapture('detection/api/frames/closed.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(round(fps / 10))
count = 0
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if count % interval == 0:
            frame_number += 1
            cv2.imwrite('detection/api/frames/tests/closed/frame_%d.png' % frame_number, frame)
        count += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()
