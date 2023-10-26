import cv2 as cv
from deepface import DeepFace

myimage = cv.imread("photos/originals.jpg")

#define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


#emotion detection model
model = DeepFace.build_model("Emotion")


#resize image using function
def resizeImg(frame, scalle=0.54):
    width = int(frame.shape[1] * scalle)
    height = int(frame.shape[0] * scalle)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)
resizedImage = resizeImg(myimage)
cv.imshow("resized", resizedImage)
cv.waitKey(0)

#convert image to greyscale
grayImage = cv.cvtColor(resizedImage, cv.COLOR_BGR2GRAY)

#load face cascade classifier
faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


#detect image using cascade
detectedFaces = faceCascade.detectMultiScale(grayImage,scaleFactor=1.1,minSize=(50,50), minNeighbors=3)

for (x,y,w,h) in detectedFaces:
    #extract face region of interest
    faceRegion = grayImage[y:y + h, x:x + w]
    #resize the input for the emotion model   
    resized_face = cv.resize(faceRegion, (48, 48), interpolation= cv.INTER_AREA)
    
    # Normalize the resized face image
    normalized_face = resized_face / 255.0

    # Reshape the image to match the input shape of the model
    reshaped_face = normalized_face.reshape(1, 48, 48, 1)

    #predict emotions using the pre-trained model
    preds = model.predict(reshaped_face)
    emotion_idx = preds.argmax()
    emotion = emotion_labels[emotion_idx]
    
    cv.rectangle(resizedImage, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.putText(resizedImage, emotion, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv.imshow("prak",resizedImage) 
cv.waitKey(0)
#cv.rectangle(resizedImage, (x,y), (x+w, y+h), (0,255,0), thickness=2)
# cv.imshow("prak",resizedImage) 
# cv.waitKey(0)
#facePart = grayImage[y:y+h, x:x+w]
# cv.imshow("prak", myimage)
# cv.imshow("gray", resizedImage)
# cv.waitKey(0)