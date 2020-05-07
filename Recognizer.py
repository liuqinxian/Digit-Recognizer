import cv2
import torch
from Recognizer_LeNet import LeNet
import numpy as np

#模型加载
net1 = LeNet()
net1.load_state_dict(torch.load('./model/LeNet.pth'))
net1.cuda()

#开启摄像头，获取摄像头参数
camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = camera.get(3)
HEIGHT = camera.get(4)
print('camera has been opened', WIDTH, HEIGHT)

#设置采样框坐标，边长
a = int(0.3 * HEIGHT)
recX = int(WIDTH / 2 - a / 2)
recY = int(HEIGHT / 2 - a / 2)

while(camera.isOpened()):

    #摄像头捕捉视频帧
    _, img = camera.read()

    #帧轮廓提取
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh[recY:recY+a, recX:recX+a]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 400:
            x, y, w, h = cv2.boundingRect(contour)

            #图片缩放为20×20
            newImg = thresh[y:y + h, x:x + w]
            if w > h:
                ww = 20
                hh = int(h * 20 / w)      
            elif w < h:
                ww = int(w * 20 / h)
                hh = 20
            else:
                ww = 20
                hh = 20
            newImg = cv2.resize(newImg, (ww, hh))

            #图片扩充为28×28
            top = int(14 - hh / 2)
            bot = 28 - hh - top
            lef = int(14 - ww / 2)
            rig = 28 - ww - lef
            newImg = cv2.copyMakeBorder(newImg, top, bot, lef, rig, cv2.BORDER_CONSTANT, value=[0,0,0])

            #图片升维
            newImg = np.expand_dims(newImg, 0)
            newImg = np.expand_dims(newImg, 0)
            newImg = np.array(newImg)
            newImg = torch.from_numpy(newImg)
            newImg = newImg.type(torch.FloatTensor)
            newImg = newImg.cuda()

            #预测
            output1 = net1.forward(newImg)
            _, num1 = torch.max(output1.data, 1)
            num1 = num1.item()

            
            

    cv2.rectangle(img, (recX, recY), (recX+a, recY+a), (255, 0, 0), 2)
    cv2.rectangle(img, (recX + x, recY + y), (recX + x + w, recY + y + h), (0, 255 ,255), 1)
    cv2.putText(img, 'LeNet : ' + str(num1), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Frame', img)
    cv2.imshow('Contours', thresh)

    k = cv2.waitKey(10)
    if k == 27:
        break

