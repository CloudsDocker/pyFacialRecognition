import cv2,sys

# the sample image file
inputImageFile=sys.argv[1]

# the HAAR cascade file, which contains the machine learned data for face detection
faceBase='haarcascade_frontalface_default.xml'

faceClassifier=cv2.CascadeClassifier(faceBase)

# use cv2 to load image file
objImage=cv2.imread(inputImageFile)

# convert the image to gray scale
cvtImage=cv2.cvtColor(objImage,cv2.COLOR_BGR2GRAY)

# to detect faces
foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.3,minNeighbors=9,minSize=(50,50),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

print(" There are {} faces in the input image".format(len(foundFaces)))

# to iterate each faces founded
for (x,y,w,h) in foundFaces:
	cv2.rectangle(objImage,(x,y),(x+w,y+h),(0,0,255),2)

#show the image
cv2.imshow("Facial Recognition Result, click anykey of keyboard to exit", objImage)
cv2.waitKey(0)