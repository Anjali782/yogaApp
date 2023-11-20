import os
from flask import Flask, request, jsonify, render_template, redirect, Response
from werkzeug.utils import secure_filename
from PIL import Image

import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2

from matplotlib import pyplot as plt
from matplotlib import image as mpimg



def get_landmark_coordinates(image_path):
    # Initialize a list to store x and y coordinates
    landmark_coordinates = []

    cap = cv2.VideoCapture(image_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Extract x and y coordinates from the landmark subset
        for landmark in landmarks:
            x, y, z = landmark.x, landmark.y, landmark.z
            landmark_coordinates.append((x, y))

    return landmark_coordinates


def calculateAngle(a,b,c):
        cc = type(a)
        #print(cc)
        x1=a[0]
        x2=b[0]
        x3=c[0]
        y1=a[1]
        y2=b[1]
        y3=c[1]
        radians = np.arctan2(y3-y2, x3-x2) - np.arctan2(y1-y2, x1-x2)
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle


        return angle

def evaluate_surya_namaskar_pose(landmarks):
    suggestions = []  # Store suggestions for improvement

    # Define the Stage enum (you should define this enum based on your needs)
    class Stage:
        one = 1
        two = 2
        three = 3
        four = 4
        five = 5
        six = 6
        seven = 7

    # Add the code for evaluating the posture based on your provided logic.

    # Calculate angles and evaluate the posture
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    print(left_elbow_angle)

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    print(right_elbow_angle)

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    print(left_knee_angle)

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    print(right_knee_angle)

    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    left_hip_s_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

    right_hip_s_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    print(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0])
    print(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0])
    print(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0])
    print(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0])

    # Add your logic to determine the stage of the Surya Namaskar pose
    pose_stage = None
    if ((pose_stage == None) and (left_elbow_angle >= 75 and left_elbow_angle <= 105) and
        (right_elbow_angle >= 75 and right_elbow_angle <= 105)):
        if (left_hip_s_angle >= 170 and left_shoulder_angle <= 190) and (right_hip_s_angle >= 170 and right_hip_s_angle <= 190):
            if (left_knee_angle >= 165 and left_knee_angle <= 195) and (right_knee_angle >= 165 and right_knee_angle <= 195):
                pose_stage = Stage.one

    elif ((pose_stage == None) and (left_elbow_angle >= 135 and right_elbow_angle >= 135) and
        (left_elbow_angle <= 180 and right_elbow_angle <= 180) and (left_knee_angle >= 150 and right_knee_angle >= 150)):
        if ((landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0] < landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0])):
            pose_stage = Stage.two

    elif ((pose_stage == None) and (left_knee_angle > 160 and right_knee_angle > 160) and
        (left_hip_s_angle < 35 and right_hip_s_angle < 35) and (left_hip_s_angle >= 5 and right_hip_s_angle >= 5)):
        if ((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]) and
            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1] < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])):
            pose_stage = Stage.three
            
            
    elif((pose_stage == None) and (left_elbow_angle>160 and right_elbow_angle>160) and (left_shoulder_angle>25 and right_shoulder_angle>25) and (left_shoulder_angle<40 and right_shoulder_angle<40)):
      if(abs(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0]-landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0])>0.40):
          pose_stage = Stage.four

    #for pose 5
    elif((pose_stage == None) and (left_knee_angle>160 and right_knee_angle>160) and (left_hip_s_angle<95 and right_hip_s_angle<95) and (left_hip_s_angle>=65 and right_hip_s_angle>=65)):
       if((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]) and (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]<landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])):
            pose_stage = Stage.five
    #for pose 6
    elif((pose_stage == None) and (left_hip_s_angle>90 and right_hip_s_angle>90) and (left_hip_s_angle<150 and right_hip_s_angle<150) and (left_knee_angle>90 and right_knee_angle>90)):
      if(landmarks[mp_pose.PoseLandmark.NOSE.value][1]>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]):
        pose_stage = Stage.six
    #for pose 7
    elif((pose_stage == None) and (left_knee_angle>150 and right_knee_angle>150) and (left_hip_s_angle>120 and right_hip_s_angle>120)):
      if((landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]) and (landmarks[mp_pose.PoseLandmark.NOSE.value][1]<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1])):
        pose_stage = Stage.seven
        
    else:
        print("Make sure you are performing surya namaskar asana")
        
    print("pose_stage------->",pose_stage)
    #suggestions = []
    if pose_stage == Stage.one:
        t = 0
        if ((left_elbow_angle>=85 and left_elbow_angle<=105) and (right_elbow_angle >=85 and right_elbow_angle<= 105)):
            t =t+1
        else:
            print("pls bring your hands towards body ")
            suggestions.append("pls bring your hands towards body ")
        if (left_hip_s_angle>=175 and left_shoulder_angle<= 185) and (right_hip_s_angle>= 175 and right_hip_s_angle<= 185):
            t = t+1
        else:
            print("Pls stand straight")
            suggestions.append("Pls stand straight")
        if (left_knee_angle >=170 and left_knee_angle <= 185 and right_knee_angle >= 170 and right_knee_angle <= 185):
            t = t+1
        else:
                print("pls dont bend your knees")
                suggestions.append("pls dont your knees")
        if(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y> landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y> landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y):
            t = t+1
        else:
            print("pls put your hands below your shoulder ")
            suggestions.append("pls put your hands below your shoulder ")
        if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x-landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x<=0.1):
            t = t+1
        else:
            print("pls put your hand together to make a NAMASKAR pose from hands")
            suggestions.append("pls put your hand together to make a NAMASKAR pose from hands")
        if t==5:
            print("you are doing great")
            suggestions.append("you are doing great")
        else:
            print("Please take a look at photo and get in right position")
            suggestions.append("Please take a look at photo and get in right position")
  
    elif pose_stage == Stage.two:
        t = 0
        if((left_elbow_angle>=140 and right_elbow_angle>=140) and (left_elbow_angle<=165 and right_elbow_angle<=165)):
            t = t+1
        else:
            print("please straighten your arms behind your head")
            suggestions.append("please straighten your arms behind your head")
        if( (left_knee_angle>=170 and right_knee_angle>=170)):
            t =t+1
        else:
            print("please don't bend your knees keep them straight")
            suggestions.append("please don't bend your knees keep them straight")
        if((landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0]>landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0]) and (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0] <landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0])):
            t =t+1
        else:
            print("please keep your wrist straight and behind your foot")
            suggestions.append("please keep your wrist straight and behind your foot")
        if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0] <landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0] <landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]):
            t =t+1
        else:
            print("please push back your hands harder")
            suggestions.append("please push back your hands harder")

        if (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0]>=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0] and landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0]>=landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0]):
            t = t+1
        else:
            print("please bend a bit more backwards ")
            suggestions.append("please bend a bit more backwards ")

        if t==5:
            # print("You are in right position ")
            suggestions.append("You are in right position ")
        else:
            print("Please take a look at photo and get in right position")
            suggestions.append("Please take a look at photo and get in right position")


    elif pose_stage == Stage.three:
        t =0
        if((left_knee_angle>170 and right_knee_angle>170)):
            t =t+1
        else:
            print("Please keep your leg straight and make your ankle touch ground")
            suggestions.append("Please keep your leg straight and make your ankle touch ground")
        if ((left_hip_s_angle<30 and right_hip_s_angle<30) and (left_hip_s_angle>=10 and right_hip_s_angle>=10)) :
            t =t+1
        else:
            print("Bend yourself properly ")
            suggestions.append("Bend yourself properly ")
        if ((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]) and (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]<landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])):
            t = t+1
        else:
            print("Your shoulder should be below your hips")
            suggestions.append("Your shoulder should be below your hips")
        if (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][1] and landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]<landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][1]):
            t =t+1
        else:
            print("please keep your waist down and in line with legs")
            suggestions.append("please keep your waist down and in line with legs")
        if ((landmarks[mp_pose.PoseLandmark.NOSE.value][1]>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]) and (landmarks[mp_pose.PoseLandmark.NOSE.value][1]>landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])):
            t =t+1
        else:
            print("look down or keep your head down")
            suggestions.append("look down or keep your head down")
        if (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]<landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1]<landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]):
            t =t+1
        else:
            print("Your right knee should touch the ground ")
            suggestions.append("Your right knee should touch the ground ")

        if t==6:
            print("you are doing great")
            suggestions.append("you are doing great")
    
    else:
        suggestions.append("Try to attempt the asana correctly it is not matching with any stage")
        

    return suggestions

UPLOAD_FOLDER = './upload_images'
app = Flask(__name__, template_folder="template")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    logger.debug('Received GET for /')
    return render_template('home.html')

@app.route('/get_array', methods=['POST'])
def get_array():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect('/')
        else:
            image = request.files['image']
            if image is not None:    
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)

                data_to_send = get_landmark_coordinates(image_path)

                if data_to_send is not None:
                    suggestions = evaluate_surya_namaskar_pose(data_to_send)
                    return render_template('result.html', suggestions=suggestions)
                else:
                    print("Error: Failed to process the image")
                    return jsonify({
                        'success': False,
                        'message': 'Failed to process the image'
                    }), 500
            else:
                print("Error: No valid image provided in the request")
                return jsonify({
                    'success': False,
                    'message': 'No valid image provided in the request'
                }), 400
    else:
        print("Error: Invalid request method")
        return jsonify({
            'success': False,
            'message': 'Invalid request method'
        }), 405
      

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




