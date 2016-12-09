# -*- coding: utf-8 -*-
import cv2,sys
import logging

# FORMAT='%(asctime)-15s %(user)-8s %(message)s'
FORMAT='%(asctime)-15s [%(levelname)s] %(lineno)d %(clientip)s %(user)-8s %(message)s'
#Full list of formats can be found at https://docs.python.org/2/library/logging.html#logging.Formatter

extraD={'clientip':'localhost','user':'todzhang'}
logging.basicConfig(filename=r'app.log',level=logging.DEBUG,format=FORMAT)
logger=logging.getLogger(__name__)

logger.debug("			start to process 		",extra=extraD)
# 使用输入的测试照片的文件名
inputImageFile=sys.argv[1]
logger.debug("	inputImageFile is %s",inputImageFile,extra=extraD)

logger.debug('使用 HAAR 的机器学习积累的原始文件，这里此文件包括了人脸识别的“经验”',extra=extraD)
faceBase='haarcascade_frontalface_default.xml'

logger.debug(' 根据机器学习库文件创建一个 classifier',extra=extraD)
faceClassifier=cv2.CascadeClassifier(faceBase)

logger.debug(' 使用库 cv2 来加载图片',extra=extraD)
objImage=cv2.imread(inputImageFile)

logger.debug(' 首先将图片进行灰度化处理，以便于进行图片分析',extra=extraD)
cvtImage=cv2.cvtColor(objImage,cv2.COLOR_BGR2GRAY)

logger.debug(' 执行detectMultiScale方法来识别物体，我们这里使用的是人脸的数据，因此用于面部识别',extra=extraD)
foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.3,minNeighbors=9,minSize=(50,50),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

logger.info(" 在图片中找到了 %d 个人脸",len(foundFaces),extra=extraD)

logger.debug(' 遍历发现的人脸',extra=extraD)
for (x,y,w,h) in foundFaces:
	cv2.rectangle(objImage,(x,y),(x+w,y+h),(0,0,255),2)

logger.debug('显示这个图片识别的结果',extra=extraD)
cv2.imshow(u'面部识别的结果已经高度框出来了。按任意键退出'.encode('gb2312'), objImage)
cv2.waitKey(0)