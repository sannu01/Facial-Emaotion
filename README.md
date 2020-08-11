# Facial-Emotion
This algorithm is especially composed of three parts, respectively, for video pretreatment,
model training and behavior recognition part. within the video preprocessing part, firstly the
first behavior of video preprocessing, using block updating background subtraction method to
realize target detection, two value image motion information is extracted, then the image input
channel convolutional neural network, through the iterative training parameters of the network,
to construct a model for convolution Behavior Recognition finally.

![facial_landmark](https://github.com/sannu01/Facial-Emotion/blob/master/output/facial_landmarks_68markup-768x619.jpg)

 EYE ASPECT RATIO (EAR):
Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you
were looking at the person), and then working clockwise around the remainder of the region:
Based on the work by Soukupová and Čech in their 2016[] paper, Real-Time Eye Blink
Detection using Facial Landmarks, we can then derive an equation that reflects this relation
called the eye aspect ratio (EAR)

![eye_aspect](https://github.com/sannu01/Facial-Emotion/blob/master/output/eye_aspect.png)

Facial Width-Height Ratio (fWHR): 
WHR is the distance measured from cheekbone to cheekbone versus the distance between the top of the lip and midbrow. fWHR are associated with traits such as aggression, risk-seeking, and egocentrism. Under the influence of the hormone, men's facial width increases in relation to height (width-height ratio or WHR), independent of body size. A high WHR is 1.9 or above. Women with high WHRs are perceived as more aggressive than average but aren't, presumably because the skull-shaping pubescent testosterone surge affects males only.

![whr](https://github.com/sannu01/Facial-Emotion/blob/master/output/facial_whr.jpg)

CNN VGG16 Model: 
This model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. The increased accuracy is obtained by replacing large kernel-sized filters with multiple small kernel-sized filters one after another. The classification model has been trained to recognize facial emotions like Anger, Sadness, Happiness, Surprise, Neutral, Fear etc

![output](https://github.com/sannu01/Facial-Emotion/blob/master/output_main.png)
