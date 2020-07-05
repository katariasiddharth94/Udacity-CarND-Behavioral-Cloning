# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---
### Files Submitted

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network model
* writeup_report.md summarizing the results
* run1.mp4 - video of track1 in autonomous mode

#### 2. Testing the Model
The project was implemented in the Udacity workspace. Run the following command in the workspace terminal and open the simulator and run track 1 in autonomous mode to test the model.'
```
python drive.py model.h5
```

Training data which was collected while driving track1 in training mode is stored in the CarND-Behavioral-Cloning-P3/new_data/ directory. 

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. This file shows the pipeline I used for creating the training data set, training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Lenet architeture. Fist, all images in the training set are normalized. In each image, the top 70 pixels includes the sky and bottom 20 pixels includes hood of car. Hence they are cropped from each image. 2 convolutional2D layers are used - the first layer has filter size of (5, 5) with depth of 6 while the second layer has filter size (5,5) with depth of 10. To each of these convolutional layers, 'relu' activation is applied to introduce non-linearity. Also, after each convolutional layer, there is a max pooling 2D layer.

The output of these layers is then flattened and fed throught 2 fully connected layers with output of size 120 and 84 respectively. This is then passed through a droput layer and finally passed through a fully connected layer to produce 1 output. The last layer has 1 output since we are trying to predict the steering angle which is a single value.

Look at the code from lines 58-75 to see the model architecture.

#### 2. Attempts to reduce overfitting in the model

A droput layer was added before the final fully connected layer with keep_prob = 0.5. This was added to prevent overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 76). The number of epochs chosen was 10 and the loss function is "mse" (mean squared error). A validation data set split of 0.2 was chosen (clone.py line 76)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The model was augmented by adding center, left and right images to the training set. Steering angles were modified for left and right images by adding/subtracting correction to original steering measurement. This helps the car steer towards the center in case it gets to either edge of the road. 

Since Track 1 consists large part of steering towards the left, the dataset was augmented by flipping each image and addingthe flipped image to the date set. Steering angle were negated for the flipped images and these new values were added to the measurements array.



