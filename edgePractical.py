import cv2 as cv
myImage = cv.imread("photos/vehicleParkingLot.jpg")
resizedImage = cv.resize(myImage,(980, 470))
edges = cv.Canny(resizedImage, 100, 500)
cv.imshow("edge Detected", edges)
# cv.imshow("resized", resizedImage)

if (cv.waitKey(0) & 0xFF == ord("q")):
    # myImage.release()
    # resizedImage.release()
    cv.destroyAllWindows()


