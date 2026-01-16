import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading

# ================== CONFIG ==================
MODEL_PATH = "front_raise_stage_model.pkl"
CAM_INDEX = 0

# ================== UTILS ==================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = joblib.load(MODEL_PATH)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# ================== SHARED STATE ==================
rep_count = 0
rep_state = "down"   # "down" or "up"
feedback = "Get into position."
done = False         # Flask will set this to stop
lock = threading.Lock()

# ================== WORKOUT LOOP ==================
def workout_loop(target_reps=None):
    global rep_count, rep_state, feedback, done

    cap = cv2.VideoCapture(CAM_INDEX)
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

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

            # ---- Feature vector (134 features) ----
            feats = []
            for lm in landmarks:
                feats.extend([lm.x, lm.y, lm.z, lm.visibility])

            # Calculate shoulder angles
            left_shoulder_angle = calculate_angle(
                [landmarks[13].x, landmarks[13].y],  # elbow
                [landmarks[11].x, landmarks[11].y],  # shoulder
                [landmarks[23].x, landmarks[23].y]   # hip
            )
            right_shoulder_angle = calculate_angle(
                [landmarks[14].x, landmarks[14].y],  # elbow
                [landmarks[12].x, landmarks[12].y],  # shoulder
                [landmarks[24].x, landmarks[24].y]   # hip
            )

            feats.extend([left_shoulder_angle, right_shoulder_angle])
            feats = np.array(feats).reshape(1, -1)

            # ---- Predict stage (optional, for debugging) ----
            stage_pred = int(model.predict(feats)[0])

            # ---- Angle-based REP COUNT LOGIC ----
            if left_shoulder_angle < 20:  # Arm down
                if rep_state == "up":  # Completed a rep
                    with lock:
                        rep_count += 1
                    rep_state = "down"
                    feedback = "Rep counted!"
                else:
                    feedback = "Raise your arm (~70°)."
            elif left_shoulder_angle > 70:  # Arm up
                if rep_state == "down":
                    rep_state = "up"
                feedback = "Lower slowly."
            else:  # Midway
                feedback = "Halfway! Control your movement."

            # ---- Draw only landmarks (no overlay) ----
            mp_drawing.draw_landmarks(frame,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

        # Show webcam feed (landmarks only, no text overlay)
        cv2.imshow('Front Raise Counter', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Stop if target reps reached
        if target_reps and rep_count >= target_reps:
            break

    cap.release()
    cv2.destroyAllWindows()
    done = True



