# Importing the Keras libraries and packages
from keras.models import model_from_json
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


