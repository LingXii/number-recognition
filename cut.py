
from PIL import Image

def binarizing(img,threshold):
    """����image������лҶȡ���ֵ����"""
    img = img.convert("L") # ת�Ҷ�
    pixdata = img.load()
    w, h = img.size
    # �����������أ�������ֵ��Ϊ��ɫ
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


def vertical(img):
    """�����ֵ�����ͼƬ���д�ֱͶӰ"""
    pixdata = img.load()
    w,h = img.size
    ver_list = []
    # ��ʼͶӰ
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x,y] == 0:
                black += 1
        ver_list.append(black)
    # �жϱ߽�
    l,r = 0,0
    flag = False
    cuts = []
    for i,count in enumerate(ver_list):
        # ��ֵ����Ϊ0
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
    temp = p.crop(n) # ����crop���������и�
    temp.save("cut%s.png" % i)