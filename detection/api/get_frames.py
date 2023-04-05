import cv2

vidcap = cv2.VideoCapture('detection/api/inclinacoes.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("detection/api/frames/image"+str(count)+".png", image)
    return hasFrames
sec = 0
frameRate = 0.1 # captura imagem a cada 1 segundo
count=1
success = getFrame(sec)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
vidcap.release()