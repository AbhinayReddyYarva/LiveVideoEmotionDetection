# LiveVideoEmotionDetection
To detect emotion in live video

Check out the below predicted output sample
![](SampleOutput/OutputGif.gif)

Updated code output
![](OutputGif.gif)

I have trained my model with Happy, Neutral and Anger. Usinging opencv haarcascade_frontalface_default.xml created the data set and trained the model. This is because to neglect the back ground and train the model with good accuracy. 
Sample images from DataSet

For Data set creation using web cam go ahead and check out https://github.com/AbhinayReddyYarva/OpenCV. We can train the same model with open source data set. When I get time, I will try to train the model with huge open source data sets and that will make the model robust.

I have trained the model TrainEmotionsCNN.py with DataSet and saved the model and weights using keras.models.model_from_json as odel in JsonModel.json and weights in JsonModel.h5 files and also in YAML as well using keras.models.model_from_yaml. 

In LiveVideoEmotionDetection.py loded the model and weights before opening the web cam. Once web cam is opened taking one frame per second and cropping the image to only face part using opencv haarcascade_frontalface_default.xml. Send the cropped image to predict the output. Below is the cropped image from video which sent to model to predict.
![](Abi.jpg)
