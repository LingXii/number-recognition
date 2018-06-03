
from PIL import Image

def binarizing(img,threshold):
    """传入image对象进行灰度、二值处理"""
    img = img.convert("L") # 转灰度
    pixdata = img.load()
    w, h = img.size
    # 遍历所有像素，大于阈值的为黑色
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


def vertical(img):
    """传入二值化后的图片进行垂直投影"""
    pixdata = img.load()
    w,h = img.size
    ver_list = []
    # 开始投影
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x,y] == 0:
                black += 1
        ver_list.append(black)
    # 判断边界
    l,r = 0,0
    flag = False
    cuts = []
    for i,count in enumerate(ver_list):
        # 阈值这里为0
        if flag is False and count > 7:
            l = i
            flag = True
        if flag and count <= 7:
            r = i-1
            flag = False
            cuts.append((l,r))
            print(l, r)
    return cuts

p = Image.open('t2_r.jpg')
b_img = binarizing(p,200)
v = vertical(b_img)

cuts = [(4,0,9,28),(18,0,25,28)]
for i,n in enumerate(cuts,1):
    temp = p.crop(n) # 调用crop函数进行切割
    temp.save("cut%s.png" % i)