import cv2,sys
import logging

FORMAT='%(asctime)-15s [%(levelname)s] %(lineno)d %(clientip)s %(user)-8s %(message)s'

extraD={'clientip':'localhost','user':'todzhang'}
logging.basicConfig(filename=r'app_en.log',level=logging.DEBUG,format=FORMAT)
logger=logging.getLogger(__name__)

logger.debug("			start to process 		",extra=extraD)
inputImageFile=sys.argv[1]
logger.debug("	inputImageFile is %s",inputImageFile,extra=extraD)

logger.debug('  we are going to make use of HAAR cascade to detect human faces',extra=extraD)
faceBase='haarcascade_frontalface_default.xml'

logger.debug(' To build one classifier based on aforeside cascade',extra=extraD)
faceClassifier=cv2.CascadeClassifier(faceBase)

logger.debug(' To load digital images via library CV2',extra=extraD)
objImage=cv2.imread(inputImageFile)

logger.debug(' Firstly, going to convert colorful image to grayscale,for further analysis',extra=extraD)
cvtImage=cv2.cvtColor(objImage,cv2.COLOR_BGR2GRAY)

logger.debug(' To call method detectMultiScale, to detect objects, here is human faces',extra=extraD)
foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.3,minNeighbors=9,minSize=(50,50),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

logger.info(" Found  %d  human faces in this image",len(foundFaces),extra=extraD)
print(" Found {} human faces in this image".format(len(foundFaces)))

logger.debug(' iterate human faces found',extra=extraD)
for (x,y,w,h) in foundFaces:
	cv2.rectangle(objImage,(x,y),(x+w,y+h),(0,0,255),2)

logger.debug('Show the result',extra=extraD)
cv2.imshow('Detected human faces highlighted. Press any key to exit. ', objImage)
cv2.waitKey(0)
