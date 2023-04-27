class Drowsy:
    def __init__(self) -> None:
        # self._eye =
        # self._mouth =
        # self._head =
        self.__list_FPF = [] # essa lista armazenará tuplas nas quais constarão o horário em segundos quando o frame foi enviado
                             # e a diferença de tempo do envio do frame anterior e o atual
        ...

    def detectDrowsiness(self, bgr_img, fps): # métodos cujos argumentos são o frame recebido em formato bgr e o valor do fps
       return bgr_img
    
    def setFPF(self, fpf):
        self.__list_FPF.append(fpf) 

    def getFPF(self):
        return self.__list_FPF
