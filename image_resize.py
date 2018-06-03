# 对将要识别的图片进行二值化、去噪、提取数字并重新约束图片大小为28x28
# 要求手写的数字连通，允许小噪点
import cv2
import queue
import numpy

# image deal
q = queue.Queue() # 广搜队列
buf = queue.Queue() # 缓冲队列

def bfs(img,vis):
    x_size = img.shape[0]
    y_size = img.shape[1]
    first = q.get()
    up_border = first[0]
    left_border = first[1]
    dowm_border = first[0]
    right_border = first[1]
    q.put(first)
    while(not q.empty()):
        p = q.get()
        buf.put(p)  # 将广搜得到的像素点放入缓冲区，等待切分
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
    return (up_border,dowm_border,left_border,right_border)


def img_resize(img):
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,75,25) # 二值化
    img= cv2.medianBlur(img,3) # 去噪
    x_size = img.shape[0]
    y_size = img.shape[1]
    visit = [[0 for y in range(y_size)] for x in range(x_size)]
    cnt=0
    for j in range(img.shape[1]):# 广搜
        for i in range(img.shape[0]):
            if (visit[i][j]==0 and img[i, j] < 30):
                visit[i][j]=1
                q.put((i,j));
                box = bfs(img,visit) # 一个框
                # 切分识别
                height = box[1]-box[0]+1
                width = box[3]-box[2]+1
                if(height*width < 300):
                    while(not buf.empty()): buf.get()
                    continue # 排除掉很小的区域，肯定是噪点
                if(height > width):
                    bias = (height - width) >> 1
                    simple_img = numpy.zeros((height,height))
                    while(not buf.empty()):
                        p = buf.get()
                        simple_img[p[0]-box[0],p[1]-box[2]+bias] = 255-img[p[0],p[1]]
                else:
                    bias = (width - height) >> 1
                    simple_img = numpy.zeros((width, width))
                    while (not buf.empty()):
                        p = buf.get()
                        simple_img[p[0]-box[0]+bias, p[1]-box[2]] = 255-img[p[0], p[1]]
                mnist_img = cv2.resize(simple_img, (28, 28))
                cv2.imwrite("./data/%s.jpg"%cnt,mnist_img)
                cnt=cnt+1

img = cv2.imread("2.jpg",0)
img_resize(img)