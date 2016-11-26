import cv2,sys
faceClassifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # the HAAR cascade file, which contains the machine learned data for face detection
objImage=cv2.imread(sys.argv[1]) # use cv2 to load image file
cvtImage=cv2.cvtColor(objImage,cv2.COLOR_BGR2GRAY) # convert the image to gray scale
foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.3,minNeighbors=9,minSize=(50,50),flags = cv2.cv.CV_HAAR_SCALE_IMAGE) # to detect faces
print(" There are {} faces in the input image".format(len(foundFaces)))
for (x,y,w,h) in foundFaces:# to iterate each faces founded
	cv2.rectangle(objImage,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow("Facial Recognition Result, click anykey of keyboard to exit", objImage) #show the image
cv2.waitKey(0)