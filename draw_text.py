import cv2
import numpy as np
from pylab import *

def draw_text(str):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    img = np.zeros((100, 300, 3), np.uint8)
    img = cv2.putText(img, str, (0, 40), font, 0.8, (255,255,255) ,2)
    # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
    cv2.imwrite("temp.jpg",img)

string = "3:0.5245381"
draw_text(string)
print("finish")