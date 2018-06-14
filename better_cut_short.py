#from itertools import groupby
from PIL import Image
#import numpy

def vertical(img):
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

def get_start_x(hist_width,r):
    mid = len(hist_width) // 2
    temp = hist_width[mid-r:mid+r]
    return mid - 4 + temp.index(min(temp))

def cut_num(path):
    p = Image.open(path)
    w,h = p.size
    r=w//8
    width_ = vertical(p)
    border = get_start_x(width_,r)
    return border

