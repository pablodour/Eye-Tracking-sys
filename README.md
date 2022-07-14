# eye_tracker_py

This module has the sole propouse of counting the pixels in each section of both eyes and then computing an average to calculate the wheels turning value for Eloflex Model F Power Wheelchair.
To start it you need to call the fllowing code

//  
    camera = cv.VideoCapture(camera number)  
    #initialize the camera  
    _, frame = camera.read()  
    speed = track_eyes(frame, delay, flag)  
//  
And you need to download the trained model from:  
https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks_GTX.dat.bz2   

Remember to clear and free the camera after the program is terminated with the following lines of code  

//  
    #closing the camera  
    camera.release()  
    #Recoder.release()  
    #closing  all the windows  
    cv.destroyAllWindows()  
//

# functions:  
## track_eyes(frame, delay, flag)  
Main function to be called in the application, the inputs:  
      -frame: frame captured from the camera  
      -delay: delay in seconds to chose when to do a prediction  
      -flag: will decide which eye/eyes(0 right, 1 left, else both) to get the reading from  
It will return the speed calculated

## landmarker(frame, face)
Function to obtain the face landmarks using an already developed model "shape_predictor_68_face_landmarks_GTX.dat"

## EyeCoordinates(eyePoints)
Function that gets the edges of the eye

## EyeCrop(frame, Eye)
Function that will return the croped eye

## EyePixels(cropedEye)
Function that first will threshold the eye (make pixels black or white) and will count the amount of blackpixels in the each region of the eye

## EyeTrack(LeftEye, RightEye, flag)
Function which will determine the speed. It counts the amount of pixels in each region, if there is a difference of 30 pixels between the sides it will start turning.
