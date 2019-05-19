# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

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

classifier.fit_generator(training_set,
                         samples_per_epoch = 320,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 107)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('DataSet/TestSet/TestHappy/TestHappy22.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Anger'
elif result[0][1] == 1:
    prediction = 'Happy'
elif result[0][2] == 1:
    prediction = 'Neutral'
print(prediction)

# Part 4  - Saving and loading the model

#Save and load the model 
from keras.models import model_from_json

# serialize model to JSON
model_json = classifier.to_json()
with open("CategoricalModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("CategoricalModel.h5")
print("Saved model to disk")

# later...
 
# load json and create model
json_file = open('CategoricalModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("CategoricalModel.h5")
print("Loaded model from disk")

# Compiling and Making New Predictions

# compile loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('DataSet/TestSet/TestHappy/TestHappy22.jpg', target_size = (64, 64))
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

# To save the model through yaml
from keras.models import model_from_yaml

# serialize model to YAML
model_yaml = classifier.to_yaml()
with open("YamlModel.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
classifier.save_weights("YamlModel.h5")
print("Saved model to disk")
 
# later...
 
# load YAML and create model
yaml_file = open('YamlModel.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_yaml_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_yaml_model.load_weights("YamlModel.h5")
print("Loaded model from disk")

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('DataSet/TestSet/TestHappy/TestHappy22.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_yaml_model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Anger'
elif result[0][1] == 1:
    prediction = 'Happy'
elif result[0][2] == 1:
    prediction = 'Neutral'
print(prediction)