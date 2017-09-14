# **Behavioral Cloning** 

---

[//]: # (Image References)

[nvidia-cnn]: doc_images/nvidia-cnn.png
[drive-straight]: doc_images/drive_straight.jpg
[drive-left]: doc_images/drive_left.jpg
[drive-right]: doc_images/drive_right.jpg
[processed-1]: doc_images/processed-image.PNG
[processed-2]: doc_images/processed-image2.PNG



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy was to use a known architecture and 
augment the data until the car was able to run one lap successfully.
Once it was able to do that, I tried adding pooling layers and removing different layers
to see how the driving behaviour changed.
This allowed me to reduce training time and possibly improve the model by reducing the number of parameters.

My first step was to use a convolutional neural network model similar to the one published by 
[Nvidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars).
I thought this model might be appropriate because they were one of the first papers 
that uses end-to-end supervised learning to control a self-driving car. 

In order to gauge how well the model was working, I split my image and steering angle data into a
training and validation set. Using the unaugmented data, I was already able to achieve low validation error.
However, the car drove off the road at some turns. 

I realized that the data needed to be augmented and pre-processed. Read Section 3 for more details.

Moreover, I added max pooling layers as suggested by several blog posts to reduce the size of the model
which shortened the training time. 

Lastly, I added dropout layers to reduce the potential of over-fitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture model consists a convolutional neural network with 3x3 and 5x5 filter sizes
and depths between 32 and 128 (steering_network.py lines 25-38). 
Max pooling layers were added after each convolutional layer.

The model uses RELU activation on every convolutional and 
fully-connected layer to introduce non-linearity (steering_network.py line 20). 
The data is normalized in the model using a Keras "BatchNormalization" layer (steering_network.py line 23).

At the end, there are four fully-connected layers with 1164, 100, 50, and 10 neurons.

Here is a visualization of the architecture by 
NVidia [[Source](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars)].


![Nvidia CNN Architecture][nvidia-cnn]

#### 3. Creation of the Training Set & Training Process

I used the training data provided by Udacity for training the model. 
In addition to the center lane data, 
I made use of the left and right driving data by adding or subtracting a small angle from their
steering angles respectively. 

Below are examples of the raw image data.

![drive-straight]

_Driving straight_


![drive-left]

_Turning left_

![drive-right]

_Turning right_

To improve the data, I first cropped the image to focus on the road and shrunk the size of image to 64x64. 
This removed the unnecessary parts of the images which reduced the number of parameters in the model
and allowed it to work better with a smaller data set.

In addition, I added a Gaussian blur over the images to reduce noise in the images.

However, I realized that the distribution of the samples was uneven. 
First, there were more left turns than right turns. In order to balance out left and right turns, 
in the generator, I flipped the images and their associated angles 50% of the time 
when the vehicle is turning.

Second, there were more straight driving than turning. To even out the distribution, I removed
75% of the straight driving samples from the data set.

Moreover, I noticed that there are shadows on the ground sometimes. To add more of those samples, 
I added tint of random darkness onto 30% of the images in the generator.

I also tried adding "salt and pepper" noise since it had worked well in the traffic sign classifier project.
However, adding those samples actually degraded the performance of the model. 
I suspected that it was because the inputs images were always from the simulator and, therefore, 
did not have noise like those in the real photos. 
Since images with noise like "salt and pepper" would never exist, 
adding those samples simply made the model worse. 

Below are examples of data after pre-processing and augmentations.

![Processed-1]
![Processed-2]


To train the model, I split the data set into training and validation sets.
The validation set helped determine if the model was over-fitting or under-fitting. 
The ideal number of epochs was 20 as I saw that the validation loss was no longer reducing.
I used an adam optimizer so that manually training the learning rate wasn't necessary.



