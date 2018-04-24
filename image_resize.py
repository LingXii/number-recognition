# 对将要识别的图片进行二值化、去噪、提取数字并重新约束图片大小为28x28
# 要求手写的数字连通，允许小噪点
import cv2
import queue
import numpy

# image deal
global max_area #用于提取数字块的全局变量
global max_width
global max_height
global max_x
global max_y
max_area = 0
max_width = 0
max_height = 0
max_x = 0
max_y = 0
q = queue.Queue() # 广搜队列

def bfs(img,vis):
    global max_area
    global max_width
    global max_height
    global max_x
    global max_y
    x_size = img.shape[0]
    y_size = img.shape[1]
    first = q.get()
    up_border = first[0]
    left_border = first[1]
    dowm_border = first[0]
    right_border = first[1]
    q.put(first)
    count=0 #像素计数
    while(not q.empty()):
        count=count+1
        p = q.get()
        if(p[0]>dowm_border): dowm_border=p[0]
        if(p[1]>right_border): right_border=p[1]
        if(p[0]<up_border):
            up_border=p[0]
            first = (p[0],first[1])
        if(p[1]<left_border):
            left_border=p[1]
            first = (first[0],p[1])
        left = (p[0] - 1, p[1])
        right = (p[0] + 1, p[1])
        up = (p[0], p[1] - 1)
        down = (p[0], p[1] + 1)
        if (left[0] >= 0 and vis[left[0]][left[1]] == 0 and img[left[0],left[1]]<30):
            q.put(left)
            vis[left[0]][left[1]] = 1
        if (right[0] < x_size and vis[right[0]][right[1]] == 0 and img[right[0],right[1]]<30):
            q.put(right)
            vis[right[0]][right[1]] = 1
        if (up[1] >= 0 and vis[up[0]][up[1]] == 0 and img[up[0],up[1]]<30):
            q.put(up)
            vis[up[0]][up[1]] = 1
        if (down[1] < y_size and vis[down[0]][down[1]] == 0 and img[down[0],down[1]]<30):
            q.put(down)
            vis[down[0]][down[1]] = 1
    if(count > max_area):
        max_area = count
        max_width = right_border-left_border+1
        max_height = dowm_border-up_border+1
        max_x = first[0]
        max_y = first[1]
    return

def img_resize(img):
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,7) # 二值化
    img= cv2.medianBlur(img,17) # 去噪
    x_size = img.shape[0]
    y_size = img.shape[1]
    visit = [[0 for y in range(y_size)] for x in range(x_size)]
    for i in range(img.shape[0]):# 广搜找最大块
        for j in range(img.shape[1]):
            if (visit[i][j]==0 and img[i, j] < 30):
                visit[i][j]=1
                q.put((i,j));
                bfs(img,visit)
    if(max_height > max_width):
        bias = (max_height - max_width) >> 1
        simple_img = numpy.zeros((max_height,max_height))
        for i in range(max_height):
            for j in range(max_width):
                simple_img[i,j+bias] = 255-img[i+max_x,j+max_y]
    else:
        bias = (max_width - max_height) >> 1
        simple_img = numpy.zeros((max_width, max_width))
        for i in range(max_height):
            for j in range(max_width):
                simple_img[i,j+bias] = 255-img[i+max_x,j+max_y]
    simple_img = cv2.resize(simple_img,(28,28))
    cv2.imwrite("1.jpg",simple_img)
    return simple_img
    # print("finish resize")