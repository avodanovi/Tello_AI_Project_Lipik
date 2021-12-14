import cv2
from djitellopy import Tello
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

w, h = 960, 720
pError = [0, 0, 0]
pid = [0.4, 0.4]

#TELLO DRONE INITIALIZATION AND TAKE OFF
def _init_dron():
    dron = Tello()
    dron.connect()
    print(dron.get_battery())
    dron.takeoff()
    dron.move_up(90)
    dron.streamon()
    return dron

#FINDING CENTER OF THE FACE AND FACE WIDTH
def find_face(results):
    # GETTING FACE LANDMARKS
    face = results.face_landmarks.landmark

    # GETTING BOUNDING BOX PARAMETERS
    cx_min = w
    cy_min = h
    cx_max = cy_max = 0
    for i in range(0, len(face)):
        cx, cy = int(face[i].x * w), int(face[i].y * h)
        if cx < cx_min:
            cx_min = cx
        if cy < cy_min:
            cy_min = cy
        if cx > cx_max:
            cx_max = cx
        if cy > cy_max:
            cy_max = cy

    # GETTING FACE CENTER
    cX = int(cx_min + (cx_max - cx_min) / 2)
    cY = int(cy_min + (cy_max - cy_min) / 2)

    # DRAWING BOUNDING BOX
    cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
    # GETTING BOUNDING BOX WIDTH
    bounding_box_width = cx_max - cx_min

    return [[cX, cY], bounding_box_width]


#SENDING COMMANDS TO DRONE TO TRACK FACE
def trackFace(info):
    # BOUNDING BOX, FACE CENTER
    box_width = info[1]
    x, y = info[0]

    print('Centar lica x:', x, ', y: ', y)
    print('Sirina lica: ', box_width)

    # DIFFERENCE BEETWEN IMAGE CENTER AND FACE CENTER IN WIDTH,HEIGHT AND FACE/IMAGE SIZE RATIO
    diff_w = (x - w // 2)
    diff_h = (y - h // 2)
    diff_z = int(w / box_width)

    # ABSOLUTE DIFFERENCE IN WIDTH, HEIGHT
    abs_diff_w = abs(diff_w)
    abs_diff_h = abs(diff_h)

    print('Udaljenost sredine lica od centra slike po sirini (diff_w): ', diff_w)
    print('Udaljenost sredine lica od centra slike po visini (diff_h): ', diff_h)
    print('Omjer sirine lica i slike (diff_z): ', diff_z)

    # GETTING SPEED FOR ALL DIRECTIONS
    upDown = pid[0] * diff_h + pid[1] * (diff_h - pError[0])
    leftRight = pid[0] * diff_w + pid[1] * (diff_w - pError[1])
    jaw = pid[0] * diff_w + pid[1] * (diff_w - pError[1])
    FrontBack = pid[0] * diff_z + pid[1] * (diff_z - pError[2])

    # UPDOWN SPEED DEPENDING ON POSITION ON SCREEN
    if abs_diff_h >= 176:
        upDown = -int(np.clip(upDown, -40, 40))
    elif (176 > abs_diff_h and abs_diff_h >= 64):
        upDown = -int(np.clip(upDown, -25, 25))
    elif abs(diff_h) < 64:
        upDown = 0

    print('Gore dolje:', upDown)

    # FRON-BACK SPEED DEPENDING ON FACE SIZE
    if diff_z >= 25:
        FrontBack = int(np.clip(FrontBack, 40, 42))
    elif ((diff_z < 25) and (diff_z > 19)):
        FrontBack = int(np.clip(FrontBack, 26, 28))
    elif ((diff_z <= 19) and (diff_z > 15)):
        FrontBack = int(np.clip(FrontBack, 18, 20))
    elif ((diff_z <= 15) and (diff_z > 13)):
        FrontBack = 0
    elif ((diff_z <= 13) and (diff_z > 10)):
        FrontBack = -int(np.clip(FrontBack, 20, 22))
    elif ((diff_z <= 10) and (diff_z >= 7)):
        FrontBack = -int(np.clip(FrontBack, 34, 36))
    elif diff_z < 7:
        FrontBack = -int(np.clip(FrontBack, 40, 42))

    print('Naprijed nazad:', FrontBack)

    # LEFT-RIGHT,JAW SPEED DEPENDING ON POSITION ON SCREEN  AND FACE SIZE
    if diff_z > 13:
        if abs_diff_w >= 350:
            leftRight = -int(np.clip(leftRight, -28, 28))
            jaw = int(np.clip(jaw, -54, 54))
        elif ((abs_diff_w >= 250) and (abs_diff_w < 350)):
            leftRight = -int(np.clip(leftRight, -24, 24))
            jaw = int(np.clip(jaw, -44, 44))
        elif ((abs_diff_w >= 190) and (abs_diff_w < 250)):
            leftRight = -int(np.clip(leftRight, -24, 24))
            jaw = int(np.clip(jaw, -34, 34))
        elif ((abs_diff_w >= 130) and (abs_diff_w < 190)):
            leftRight = int(np.clip(leftRight, -22, 22))
            jaw = int(np.clip(jaw, -24, 24))
        elif ((abs_diff_w > 64) and (abs_diff_w < 130)):
            leftRight = int(np.clip(leftRight, -13, 13))
            jaw = int(np.clip(jaw, -18, 18))
        elif abs_diff_w <= 64:
            leftRight = 0
            jaw = 0
    elif ((diff_z <= 13) and (diff_z > 4)):
        if abs_diff_w >= 300:
            leftRight = -int(np.clip(leftRight, -24, 24))
            jaw = int(np.clip(jaw, -52, 52))
        elif ((abs_diff_w >= 200) and (abs_diff_w < 300)):
            leftRight = -int(np.clip(leftRight, -20, 20))
            jaw = int(np.clip(jaw, -42, 42))
        elif ((abs_diff_w >= 140) and (abs_diff_w < 200)):
            leftRight = -int(np.clip(leftRight, -18, 18))
            jaw = int(np.clip(jaw, -32, 32))
        elif ((abs_diff_w >= 90) and (abs_diff_w < 140)):
            leftRight = int(np.clip(leftRight, -14, 14))
            jaw = int(np.clip(jaw, -28, 28))
        elif ((abs_diff_w > 40) and (abs_diff_w < 90)):
            leftRight = int(np.clip(leftRight, -10, 10))
            jaw = int(np.clip(jaw, -22, 22))
        elif abs_diff_w <= 40:
            leftRight = 0
            jaw = 0
    if diff_z <= 4:
        if abs_diff_w <= 100:
            leftRight = 0
            jaw = 0
        elif ((abs_diff_w > 100) and (abs_diff_w <= 200)):
            leftRight = int(np.clip(leftRight, -16, 16))
            jaw = int(np.clip(jaw, -12, 12))
        elif ((abs_diff_w > 200) and (abs_diff_w <= 280)):
            leftRight = int(np.clip(leftRight, -20, 20))
            jaw = int(np.clip(jaw, -14, 14))
        elif abs_diff_w > 280:
            leftRight = int(np.clip(leftRight, -28, 28))
            jaw = -int(np.clip(jaw, -16, 16))

    dron.send_rc_control(leftRight, FrontBack, upDown, jaw)

    return [diff_h, diff_w, diff_z]

def get_gesture(results, model1, model2):
    hand = results.right_hand_landmarks.landmark
    row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand]).flatten())

    X = pd.DataFrame([row])
    hand_language_class1 = model1.predict(X)[0]
    hand_language_class2 = model2.predict(X)[0]
    if hand_language_class2 != hand_language_class1:
        hand_language_class = "NONE"
    else:
        hand_language_class = hand_language_class1
    return hand_language_class

def recognize_gestures(gesture):
    front_back_velocity = up_down_velocity = left_right_velocity = yaw_velocity = 0
    if gesture == "FRONT":  # Forward
        front_back_velocity = 30
        up_down_velocity = left_right_velocity = yaw_velocity = 0
    elif gesture == "BACK":  # Back
        front_back_velocity = -30
        up_down_velocity = left_right_velocity = yaw_velocity = 0
    elif gesture == "UP":  # UP
        up_down_velocity = 30
        front_back_velocity = left_right_velocity = yaw_velocity = 0
    elif gesture == "DOWN":  # DOWN
        up_down_velocity = -30
        front_back_velocity = left_right_velocity = yaw_velocity = 0
    elif gesture == "LEFT":  # LEFT
        left_right_velocity = 30
        front_back_velocity = up_down_velocity = yaw_velocity = 0
    elif gesture == "RIGHT":  # RIGHT
        left_right_velocity = -30
        front_back_velocity = up_down_velocity = yaw_velocity = 0
    elif ((gesture == "FLIP") and (dron.get_battery() > 60)):
        dron.flip_forward()
        front_back_velocity = up_down_velocity = left_right_velocity = yaw_velocity = 0

    dron.send_rc_control(left_right_velocity, front_back_velocity, up_down_velocity, yaw_velocity)

dron = _init_dron()

with open('hand_language_lr_final.pkl', 'rb') as f:
    model1 = pickle.load(f)
with open('hand_language_rc_final.pkl', 'rb') as f:
    model2 = pickle.load(f)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        # GETTING IMAGE FROM DRONE
        img = dron.get_frame_read().frame

        # PREPARING IMAGE
        img = cv2.resize(img, (w, h))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # GETTING LANDMARK RESULTS
        results = holistic.process(image)

        # RETURNING PICTURE TO NORMAL
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # DRAWING LANDMARKS ON IMAGE
        # FACE LANDMARKS
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # HAND LANDMARKS
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        gesture = "NONE"

        try:
            gesture = get_gesture(results, model1, model2)
        except:
            gesture = "NONE"

        if gesture == "NONE":
            try:
                info = find_face(results)
                pError = trackFace(info)
            except:
                print('Didnt find face. Moving stop.')
                dron.send_rc_control(0, 0, 0, 0)
        else:
            recognize_gestures(gesture)

        cv2.imshow('Output', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            dron.streamoff()
            dron.land()
            break

cv2.destroyAllWindows()
