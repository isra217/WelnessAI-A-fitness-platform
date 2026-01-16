import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
from collections import deque

# ================== CONFIG ==================
MODEL_PATH = "shoulder_press_stage_model.pkl"
CAM_INDEX = 0               # default webcam
STABILITY_FRAMES = 3        # stable frames to confirm stage
ANGLE_THRESH = 30           # min ROM to count a rep

# ================== SETUP ==================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = joblib.load(MODEL_PATH)

def calculate_angle(a, b, c):
    """Calculate angle between 3 points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# ================== SHARED STATE ==================
rep_count = 0
feedback = "Get into starting position."
done = False
lock = threading.Lock()

# ================== WORKOUT LOOP ==================
def workout_loop(target_reps=None):
    global rep_count, feedback, done

    cap = cv2.VideoCapture(CAM_INDEX)
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    stage_history = deque(maxlen=STABILITY_FRAMES)
    rep_started = False
    min_angle = 180
    max_angle = 0

    while cap.isOpened() and not done:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Left arm keypoints
            l_sh = [landmarks[11].x, landmarks[11].y]  # LEFT_SHOULDER
            l_el = [landmarks[13].x, landmarks[13].y]  # LEFT_ELBOW
            l_hip = [landmarks[23].x, landmarks[23].y] # LEFT_HIP

            left_shoulder_angle = calculate_angle(l_el, l_sh, l_hip)

            # Prepare features for model (match training: 132 keypoints + 1 angle)
            feats = []
            for lm in landmarks:
                feats.extend([lm.x, lm.y, lm.z, lm.visibility])
            feats.append(left_shoulder_angle)  # only 1 angle
            feats = np.array(feats).reshape(1, -1)

            # Predict stage
            pred = int(model.predict(feats)[0])
            stage_history.append(pred)

            # Stable stage detection
            if len(stage_history) == STABILITY_FRAMES and len(set(stage_history)) == 1:
                stable_stage = stage_history[0]

                # ---- REP LOGIC ----
                if stable_stage == 1 and not rep_started:
                    # Arm up, start rep
                    rep_started = True
                    min_angle = left_shoulder_angle
                    max_angle = left_shoulder_angle

                elif stable_stage == 0 and rep_started:
                    # Arm down, complete rep
                    if (max_angle - min_angle) > ANGLE_THRESH:
                        with lock:
                            rep_count += 1
                        feedback = "Rep counted!"
                    rep_started = False
                    min_angle = 180
                    max_angle = 0

                # Update min/max during rep
                if rep_started:
                    min_angle = min(min_angle, left_shoulder_angle)
                    max_angle = max(max_angle, left_shoulder_angle)

                # ---- FEEDBACK ----
                if stable_stage == 1:
                    feedback = "Good! Lower your arm slowly."
                elif stable_stage == 0:
                    feedback = "Raise your arm up."
                elif stable_stage == 2:
                    feedback = "Halfway! Control your movement."
                else:
                    feedback = "Adjust your position."

            # ---- Draw landmarks only ----
            mp_drawing.draw_landmarks(frame,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

        # Show webcam feed (landmarks only, no overlay)
        cv2.imshow('Shoulder Press Counter', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Stop if target reps reached
        if target_reps and rep_count >= target_reps:
            break

    cap.release()
    cv2.destroyAllWindows()
    done = True





