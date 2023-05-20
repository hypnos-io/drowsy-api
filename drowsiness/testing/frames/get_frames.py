import cv2

# Open the video file
cap = cv2.VideoCapture(r'test_tired_2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
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
                cv2.imwrite('frame_00%d.png' % frame_number, frame)
            elif 10 <= frame_number <= 100:
                cv2.imwrite('frame_0%d.png' % frame_number, frame)
            else:
                cv2.imwrite('frame_%d.png' % frame_number, frame)
            
        count += 1
    else:
        break

cap.release()