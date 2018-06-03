from PIL import Image

i = 1
j = 1
img = Image.open("10.jpg")#读取系统的内照片

def the_color(i,j,k):
    if (i>=50 and i<=170 and j>=60 and j<=185 and k>=100 and k<=225):
        return 1
    else:
        return 0
        


width = img.size[0]#长度
height = img.size[1]#宽度
for i in range(0,height):#遍历所有长度的点
    for j in range(0,width):#遍历所有宽度的点
        data = (img.getpixel((j,i)))

        if (the_color(data[0],data[1],data[2])):
            if (j<width-15):
                for l in range(0,15):
                    data = (img.getpixel((j+l,i)))
                    if(not the_color(data[0],data[1],data[2])):
                        break
                if(l==14):
                    for l in range(0,15):
                        img.putpixel((j+l,i),(255,255,255,255))
                j=j+l
            else:
                img.putpixel((j,i),(255,255,255,255))
            
img = img.convert("RGB")#把图片强制转成RGB
img.save("1000000.jpg")#保存修改像素点后的图片
