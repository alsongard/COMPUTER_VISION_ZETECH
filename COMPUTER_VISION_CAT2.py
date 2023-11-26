import cv2 as cv
import numpy as nm

#detect image
myimage = cv.imread("photos/manyCarImages.jpg")

#resize image
def resizedImage(frame, scalle=0.59):
    width = int(frame.shape[1]* scalle)
    height = int(frame.shape[0]* scalle)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

#cv.imshow("new", myimage)
#CHECK RESIZED IMAGE

resizedImage = resizedImage(myimage)

cv.imshow("new", resizedImage)
# convert to grayScale

grayImage = cv.cvtColor(resizedImage, cv.COLOR_BGR2GRAY)

# cv.imshow("gray", grayImage)
loadClassifier = cv.CascadeClassifier("cars.xml")




cars = loadClassifier.detectMultiScale(grayImage, scaleFactor=1.1,minSize=[50,50], minNeighbors=1)
for (x,y,w,h) in cars:
    cv.rectangle(grayImage,(x,y), (x+w, y+h), (0,0,255),5)
cv.imshow("Result",grayImage)
cv.waitKey(0)
if (0xFF == ord("q")):
    cv.destroyAllWindow()