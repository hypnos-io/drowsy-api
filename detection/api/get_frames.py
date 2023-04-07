import cv2

vidcap = cv2.VideoCapture('detection/api/teste.mp4')
def get_frame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    has_frames, image = vidcap.read()
    if has_frames:
        cv2.imwrite("detection/api/frames/warning/image"+str(count)+".png", image)
    return has_frames
sec = 0
frame_rate = 0.1 # captura imagem a cada 1 segundo
count=1
success = get_frame(sec)
while success:
    count = count + 1
    sec = sec + frame_rate
    sec = round(sec, 2)
    success = get_frame(sec)
    
vidcap.release()