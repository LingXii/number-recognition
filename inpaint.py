 # encoding: utf-8    
import cv2
from matplotlib import pyplot as plt

src = cv2.imread("4.jpg", 0)
src = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 165, 15)
cv2.imwrite('tmp1.jpg', src)
mask = cv2.imread('black.png', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('111.jpg', src)

dst = cv2.inpaint(src, mask, 3, cv2.INPAINT_TELEA)
cv2.imwrite('afterinpaint.png', dst)
print("inpaint done")

img = cv2.imread('afterinpaint.png')

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)


cv2.imwrite('112.jpg', dst)
print("yeah")