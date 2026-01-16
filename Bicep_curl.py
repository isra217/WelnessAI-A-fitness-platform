import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import threading

# ================== config ==================
MODEL_PATH = "bicep_curl_stage_model.pkl"   # trained model
CAM_INDEX = 0                               # 0 = webcam
STABILITY_FRAMES = 3                        # frames required for stable stage
ANGLE_THRESH = 30                           # min ROM difference to count a rep

# ================== utils ==================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# load trained model
model = joblib.load(MODEL_PATH)

def calculate_angle(a, b, c):
    """Calculate angle between 3 points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# ================== shared state ==================
rep_count = 0
feedback = "Get into position to start."
done = False   # <-- Flask will set this when stopping
lock = threading.Lock()

# ================== main workout loop ==================
def workout_loop(target_reps=None):
    global rep_count, feedback, done

    cap = cv2.VideoCapture(CAM_INDEX)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    last_stable_stage = None
    stable_stage = None
    stage_history = deque(maxlen=STABILITY_FRAMES)

    min_elbow_angle = 180
    max_elbow_angle = 0
    rep_started = False

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

            feats = []
            for lm in landmarks:
                feats.extend([lm.x, lm.y, lm.z, lm.visibility])

            # calculate angles
            l_sh = [landmarks[11].x, landmarks[11].y]
            l_el = [landmarks[13].x, landmarks[13].y]
            l_wr = [landmarks[15].x, landmarks[15].y]
            r_sh = [landmarks[12].x, landmarks[12].y]
            r_el = [landmarks[14].x, landmarks[14].y]
            r_wr = [landmarks[16].x, landmarks[16].y]

            left_angle = calculate_angle(l_sh, l_el, l_wr)
            right_angle = calculate_angle(r_sh, r_el, r_wr)
            l_shoulder_angle = calculate_angle([landmarks[23].x, landmarks[23].y], l_sh, l_el)
            r_shoulder_angle = calculate_angle([landmarks[24].x, landmarks[24].y], r_sh, r_el)

            feats.extend([left_angle, right_angle, l_shoulder_angle, r_shoulder_angle])
            feats = np.array(feats).reshape(1, -1)

            pred = int(model.predict(feats)[0])
            stage_history.append(pred)

            if len(stage_history) == STABILITY_FRAMES and len(set(stage_history)) == 1:
                stable_stage = stage_history[0]

                min_elbow_angle = min(min_elbow_angle, left_angle)
                max_elbow_angle = max(max_elbow_angle, left_angle)

                if not rep_started and last_stable_stage in [0, 2] and stable_stage == 1:
                    rep_started = True
                    min_elbow_angle = left_angle
                    max_elbow_angle = left_angle

                elif rep_started and last_stable_stage in [1, 2] and stable_stage == 0:
                    if (max_elbow_angle - min_elbow_angle) > ANGLE_THRESH:
                        with lock:
                            rep_count += 1
                        print(f"Rep counted! Total: {rep_count}")
                    rep_started = False

                # ---- feedback logic ----
                if stable_stage == 1:
                    feedback = "Good! Now lower your arm slowly (~60°)."
                elif stable_stage == 0:
                    feedback = "Nice! Push back up (~170°)."
                elif stable_stage == 2:
                    feedback = "Halfway there! Keep control."
                else:
                    feedback = "Adjust your position."

                last_stable_stage = stable_stage

            # Draw landmarks (keep webcam visualization)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ✅ No overlay of reps/feedback on webcam (browser handles it)
        cv2.imshow('Bicep Curl Counter', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # stop if reached target reps
        if target_reps and rep_count >= target_reps:
            break

    cap.release()
    cv2.destroyAllWindows()
    done = True



