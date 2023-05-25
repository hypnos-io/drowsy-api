import cv2 as cv

# Open the video file
cap = cv.VideoCapture(r'drowsiness\testing\tired_0.mp4')
fps = cap.get(cv.CAP_PROP_FPS)
interval = int(round(fps / 10))
count = 0
frame_number = 0
prefix = ''

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if count % interval == 0:
            frame_number += 1
            if frame_number < 10:
                cv.imwrite(r'drowsiness\testing\frames\frame_00%d.png' % frame_number, frame)
            elif 10 <= frame_number < 100:
                cv.imwrite(r'drowsiness\testing\frames\frame_0%d.png' % frame_number, frame)
            else:
                cv.imwrite(r'drowsiness\testing\frames\frame_%d.png' % frame_number, frame)
            
        count += 1
    else:
        break

cap.release()