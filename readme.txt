Tello_AI_Project

Libraries:

mediapipe(version 0.8.7.3)
 -for face and hand recognition/tracking

djitellopy(version(2.4.0)
 -sending commands to Tello drone

opencv(version 4.5.3.56) 
 - image processing

pandas(version 1.3.3)
 - pandas dataframes for data manipulation

scikit-learn(version 0.24.2)
 -machine learning algorihms
 
numpy
 -numpy arrays
csv
 -saving data in csv file

pickle
 -packing machine learning models


#Dataset.py

In this file we are making a dataset for a hand gesture controller.
Before starting  put a drone in a good position, on a height of 2,3 m.
Make sure a camera has a clear sight.
Put in variable "class_name" name of the gesture you want to collect data for.
For example 6 gestures is made(UP,DOWN,RIGHT,LEFT,FRONT,BACK,FLIP).
You can make your own gestures dataset.
Stand in front of a drone camera with the right fist in a gesture you want to save
in a dataset. Start the script and walk around in front of the camera.
With the medipipe's mp.holistic.Holistic module face and hand landmarks are saved in 
"results" variable.
Each landmark poseses 4 values:
 x - height of a landmark in image
 y- width of a landmark in image
 z -depth of a landmark in picture depending on other landmarks
 visibility 

In varibale "pose"  only right hand landmarks are being saved.
These values are saved in a csv file(in example "GesturesFinal.csv").
It is important to get the cordinates of a gesture in many different angles so
walk around in front of the camera.
Repeat the process for every gesture you want in the dataset.


#Model_train.py

After collecting the right landmark values in a csv file 
we can train the model for the hand recognition controller.
Csv file (in example "Gestures_final.csv") is saved in pandas dataframe.
Using scikit_learns module train_test_split dataframe is split in a
test and train subsets.
Four diferent machine learning algorithms are used  to get 4 different models
(Logistic regression, ridge, random forest, gradient boost).
After checking the accuracy_score  models are saved using pickle library.
Two models with best accuracy_score will be used for a gesture controll.


#Telo_tracking_gestures_final



Functions:

init_dron()
 -initializing drone
 -take off
 -starting the video stream

find_face(results)
 - "results" variable contains landmark values 
 -finding lowest/heighest width/height cordinates (x,y) from all landmarks in the image.
  (cx_min,cx_max,cy_min,_cy_max)
 -calculateing center of the face(cX,CY) and face width(bounding_box_width).
 -returning center of the face and face width

track_face(info)
 -"info" contains cordinates of center of the face and face width
 -calculating diffrence in height/width in an image between center of the face and center of the image( diff_w, diff_h)
 -claculating relationship between face width and image width(diff_z)
 -depending on calculated values sending commands to a Tello drone( leftRight, FrontBack, upDown, jaw)
 -Pid algorithm is used for correcting the difference in values send to the drone each time function is called

get_gestures(results,model1,model2)
 -"results" variable contains landmark values 
 -"model1","model2" are two best models from Model_train.py
 -using models to predict gesture depending on the values of the right hand landmarks on the image 
 -if both models predicted the same gesture return gesture's name


recognize_gestures(gesture)
 -gesture contains name of the gesture(from get_gestures function)
 -depending on the name of the gesture send commands to a drone
  
 

#main
 -initializing Tello drone
 -loading models for hand controll
 -saveing face and hand landmarks from current image in the video stream
  (using mediapipe's mp.solutions.holistic module) in "results" variable
 -drawing landmarks on image(using medipipe's mp.solutions.drawing_utils)
 -checking if gesture is recognized in the current image
 -if gesture is recognized calling recognize_gesture function to send commands to the drone
 -if there is no gestures recognized sending "results" from mp.solutions.holistic processing to 
  find_face function
 -find_face functions returns cordinates of center of the face and face width 
 -sending center of the face and face width("info") to the trackFace function so it can
  send coresponding commands to the drone
  

 
 

  
 