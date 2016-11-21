# -*- coding: utf-8 -*-
import cv2,sys

# 使用输入的测试照片的文件名
inputImageFile=sys.argv[1]

# 使用 HAAR 的机器学习积累的原始文件，这里此文件包括了人脸识别的“经验”
faceBase='haarcascade_frontalface_default.xml'

# 根据机器学习库文件创建一个 classifier
faceClassifier=cv2.CascadeClassifier(faceBase)

# 使用库 cv2 来加载图片
objImage=cv2.imread(inputImageFile)

# 首先将图片进行灰度化处理，以便于进行图片分析
cvtImage=cv2.cvtColor(objImage,cv2.COLOR_BGR2GRAY)

# 执行detectMultiScale方法来识别物体，我们这里使用的是人脸的数据，因此用于面部识别
foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.3,minNeighbors=9,minSize=(50,50),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

print(" 在图片中找到了 {} 个人脸".format(len(foundFaces)))

# 遍历发现的人脸
for (x,y,w,h) in foundFaces:
	cv2.rectangle(objImage,(x,y),(x+w,y+h),(0,0,255),2)

#显示这个图片识别的结果
cv2.imshow(u'面部识别的结果已经高度框出来了。按任意键退出'.encode('gb2312'), objImage)
cv2.waitKey(0)