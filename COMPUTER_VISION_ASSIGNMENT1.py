#import opencv2 
import cv2 as cv

#create funt to resize image
def resizeImg(frame, scalle=0.16):
    width = int(frame.shape[1] * scalle)
    height = int(frame.shape[0] * scalle)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

#provide path to image location
myimage = cv.imread("photos/image.jpg")
#resize image using  resizeImg func
new_image = resizeImg(myimage)
# cv.imshow("resized picture", new_image)
# cv.imshow("picture", myimage)

#inorder to detect a face in an img we convert to gray image
gray_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
#display image
cv.imshow("Grey Window", gray_image)

#Cascade classifier haar_face to detect face in img
haar_cascade = cv.CascadeClassifier("haar_face.xml")
faces_detect = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2)
print(f'the number of faces are {len(faces_detect)}')
#draw box around face
for (x,y,w,h) in faces_detect:
    scanned_image = cv.rectangle(new_image, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    cv.imshow("detect Faces", scanned_image)
    cv.waitKey(0)

if 0xFF== ord('d'):    
    cv.destroyAllWindows()

    
"""
 ageProto = "age_deploy.prototxt"
 ageModel = "age_net.caffemodel"
 genderProto = "gender_deploy.prototxt"
 genderModel = "gender_net.caffemodel"

 ageNet = cv.dnn.readNet(ageModel, ageProto)
 genderNet = cv.dnn.readNet(genderModel, genderProto)


 ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
 genderList = ['Male', 'Female']
 MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

 for bbox in faces_detect:
     face = scanned_image[bbox[1]:bbox[2],bbox[0],bbox[2]]
     blob = cv.dnn.blobFromImage(face, 1.0, (277,277), MODEL_MEAN_VALUES, swapRB=False)
     genderNet.setInput(blob)
     genderPred = genderNet.forward()
     gender = genderList[genderPred[0].argmax()]

     ageNet.setInput(blob)
     agePred = ageNet.forward()
     age = ageList[agePred[0].argmax()]

     label = "{},{}" .format(gender, age)
     cv.putText(face, label, (bbox[0], bbox[1] - 10, cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 2))

"""


