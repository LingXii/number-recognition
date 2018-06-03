#from itertools import groupby
from PIL import Image
#import numpy

def vertical(img):
    """�����ֵ�����ͼƬ���д�ֱͶӰ"""
    pixdata = img.load()
    w,h = img.size
    result = []
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x,y] == 255:
                black += 1
        result.append(black)
    return result

def get_start_x(hist_width):
    """����ͼƬ��ֱͶӰ�Ľ����ȷ�����
       hist_width�м�ֵ ǰ��ȡ4��ֵ ���ⷶΧ��ȡ��Сֵ
    """
    mid = len(hist_width) // 2 # ע��py3 ������py2��ͬ
    temp = hist_width[mid-4:mid+5]
    return mid - 4 + temp.index(min(temp))


if __name__ == '__main__':
    p = Image.open("5.jpg")
    p = p.resize((28, 28), Image.ANTIALIAS)

    width_ = vertical(p)
    border = get_start_x(width_)
    cuts = [(0,0,border,28),(border,0,28,28)]
    for i,n in enumerate(cuts,1):
        tmp = p.crop(n)
        tmp.save("7better_cut_short%s.png"%i)
