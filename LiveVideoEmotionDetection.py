# Importing the Keras libraries and packages
import cv2
import math

#from google.colab.patches import cv2_imshow
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
font = cv2.FONT_HERSHEY_SIMPLEX

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('DataSet/TrainSet',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('DataSet/TestSet',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
 
# load json and create model
json_file = open('JsonModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("JsonModel.h5")
print("Loaded model from disk")

# Compiling and Making New Predictions

# compile loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

videoname = 'video.avi'
frames_per_second = 24
videotype = cv2.VideoWriter_fourcc(*'XVID') #fourcc
out = cv2.VideoWriter(videoname, videotype, 20, (640, 480))

cap = cv2.VideoCapture(0)
frameRate = cap.get(5)
print(frameRate)

while True:
    frameId = cap.get(1) #current frame number--this is for frame rate
    ret, img = cap.read()
    #show the mark on face
    if (frameId % (math.floor(frameRate/30)) == 0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)  #2 is width of the line 
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            crop_img = img[y:y+h, x:x+w]
        
        #pridicting with model 
        #converting ndarray to image
        #test_image = Image.fromarray(img, 'RGB')
        filename ="Abi.jpg"
        cv2.imwrite(filename, crop_img)
        cv2.imshow('crop_img',crop_img)
        #with open('vijay.jpg', 'wb') as f:
            #f.write(img)
        test_image = image.load_img("Abi.jpg", target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'Anger'
        elif result[0][1] == 1:
            prediction = 'Happy'
        elif result[0][2] == 1:
            prediction = 'Neutral'
        print(prediction)
    
        #Write text on the rectanguler box.
        cv2.putText(img, prediction, (286, 104), font, 0.8, (0, 0, 255), 3, cv2.LINE_AA)
        out.write(img)
        
    cv2.imshow('img',img)
    #cv2.imshow("gray",gray)
  
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()