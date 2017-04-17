#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./img/center.jpg "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./img/left.jpg    "Recovery Image"
[image4]: ./img/center_angle.jpg "Recovery Image"
[image5]: ./img/right.jpg "Recovery Image"
[image6]: ./img/flip_init.jpg "Normal Image"
[image7]: ./img/flip.jpg "Flipped Image"


###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 64 (model.py lines 133-140) followed by a full connected layers. 

The model includes ELU layers to introduce nonlinearity (code line 134), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains after each convolutional layer a max polling layer and the full connected layers are followed by dropout layers in order to reduce overfitting (model.py lines 132). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 38-40). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 159).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road and also to enable the recovery of the car. I used a combination betweend the center lane driving and  recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to first use a simple architecture as LeNet
and see how the model is performing. However by using this architecture the data was underfitted.

I then tried a more evolved network as the end-to-end NVIDIA architecture. With only modifing the layer sizes this network had a higher 
mean squared error so I decided to go back to LeNet and gradually add new convolution layers and full conected layers until the model was overfitting the data.

To combat the overfitting, I first introduced dropouts layers after the fully connected layers and so the training and validation loss were of the same order.
Then I experimented with dropouts and max pooling layers after the convolutional layers and obeserved that only by adding the 
max polling layers were yield better results.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, 
I used augmented images that were randomly shifted and randomly. Also the images from the left and right camera were used and
a steering angle needed for recovery was calculated. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 72-116) consisted of a convolution neural network with the following layers and layer sizes 

|  Layers |  Size |
|---|---|
| Lambda   |   ( 64, 64, 3) |
|Conv2D |    (64, 64, 32)   
|MaxPooling | (32, 32, 32) 
| Conv2D  |  (30, 30, 32) |
|  MaxPooling | (15,15,32)  |
|  Relu |    (15, 15, 32)
|   Conv2D |  (13, 13, 64) |
|  MaxPooling |  (6, 6, 64) |
|  Relu |  (6, 6, 64) |
|  Flatten |  2304 |
|  Dense | 128  |
|  Dropout(0.5) |   |
|  Dense | 64  |
|  Dropout(0.5) |  |
|  Dense | 1  |


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I used the left and right camera images to simulate the effect of car wandering off to the side, and recovering.
The angle needed for recovery was calculated by adding a small angle .25 to the left camera and subtracting a small angle of 0.25 from the right camera. 
These images show what a recovery looks like starting from :

Angle : 0.25

![alt text][image3] 

Angle : 0

![alt text][image4]

Angle : -0.25

![alt text][image5]


To augment the data set, I also flipped images and angles thinking that this would help for helping with the left turn bias. 
For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]

I used a generator and I randomly choosed between the right, middle and left camera and then augmented the data as above. 
In that way I could generate a random number of images.I preprocessed this data by resizing it to 64X64 pixels, 
cropping the image with 1/5 from the top and 25 pixels from the bottom and normalized the image 


I finally randomly shuffled the data set and putting the last 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6. I used an adam optimizer so that manually training the learning rate wasn't necessary.
