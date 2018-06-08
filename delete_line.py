# 对输入图片进行二值化、去线操作
import cv2
import queue
import numpy

q = queue.Queue() # 广搜队列

def bfs(img,vis,line_width):
    x_size = img.shape[0]
    y_size = img.shape[1]
    while(not q.empty()):
        p = q.get()
        img[p[0],p[1]] = 128
        # buf.put(p)  # 将广搜得到的像素点放入缓冲区，等待切分
        right = [(p[0], p[1]+1),(p[0], p[1]+3),(p[0], p[1]+5),(p[0], p[1]+7),(p[0], p[1]+9)]
        up = (p[0]-1, p[1]+1)
        down = (p[0]+1, p[1]+1)
        for i in range(1):
            if (right[i][1] < y_size and vis[right[i][0]][right[i][1]] == 0 and img[right[i][0], right[i][1]] < 30):
                q.put(right[i])
                vis[right[i][0]][right[i][1]] = 1
        if (p[1]%35==0 and up[0] >= 0 and up[1]<y_size and vis[up[0]][up[1]] == 0 and img[up[0],up[1]]<30):
            up_st = up[0]+line_width
            if((up_st<x_size and img[up_st,up[1]]) or up_st>=x_size):
                q.put(up)
                vis[up[0]][up[1]] = 1
        if (p[1]%35==0 and down[0] < x_size and down[1]<y_size and vis[down[0]][down[1]] == 0 and img[down[0],down[1]]<30):
            down_st = down[0]-line_width
            if ((down_st < x_size and img[down_st, up[1]]) or down_st >= x_size):
                q.put(down)
                vis[down[0]][down[1]] = 1
    return img

def del_line(img):
    global line_width;
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 165, 15)  # 二值化
    img = cv2.medianBlur(img, 3)  # 去噪
    # cv2.imwrite("temp0.jpg", img)
    x_size = img.shape[0]
    y_size = img.shape[1]
    visit = [[0 for y in range(y_size)] for x in range(x_size)]
    line_count = 0;
    sum = 0;
    for i in range(250,img.shape[0]):
        if (visit[i][425]==0 and img[i, 425] < 30):
            sum = sum+1
            if(i>0 and img[i,425]!=img[i-1,425]): line_count = line_count+1
            visit[i][425]=1
            q.put((i,425))
    line_width = sum//line_count+2
    img = bfs(img,visit,line_width)
    img = cv2.medianBlur(img, 3)  # 去噪
    kernel = numpy.uint8(numpy.zeros((5, 5)))
    for x in range(5):
        kernel[x, 2] = 1;
        kernel[2, x] = 1;
    img = cv2.erode(img, kernel)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    cv2.imwrite("temp.jpg",img)
    print("finish delete line")
    return img #返回一张图片，干扰线将用灰色（128）标出

# img = cv2.imread("T8.jpg",0)
# del_line(img)