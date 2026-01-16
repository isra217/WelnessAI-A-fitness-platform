from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import importlib

app = Flask(__name__)
CORS(app)

# ================== GLOBAL STATE ==================
exercise_thread = None
current_exercise = None
exercise_module_obj = None  # <-- keep reference to running module
lock = threading.Lock()
TARGET_REPS = 10

# ================== MAP EXERCISE NAMES TO FILES ==================
EXERCISE_MODULES = {
    "Bicep_curl": "Bicep_curl",
    "front_raise": "front_raise",
    "shoulder_press": "shoulder_press"
}

# ================== RUN EXERCISE ==================
def run_exercise(exercise_name):
    global exercise_module_obj

    module_name = EXERCISE_MODULES.get(exercise_name)
    if not module_name:
        return

    # Dynamically import exercise module
    exercise_module_obj = importlib.import_module(module_name)

    # Reset module globals safely
    with lock:
        exercise_module_obj.rep_count = 0
        exercise_module_obj.feedback = "Get into starting position."
        exercise_module_obj.done = False

    # Run the workout loop (this opens the webcam automatically)
    exercise_module_obj.workout_loop(target_reps=TARGET_REPS)

# ================== API ==================
@app.route("/start", methods=["POST"])
def start_exercise():
    global exercise_thread, current_exercise

    data = request.get_json()
    exercise_type = data.get("exercise")

    if exercise_type not in EXERCISE_MODULES:
        return jsonify({"status": "error", "message": "Invalid exercise type."})

    if exercise_thread and exercise_thread.is_alive():
        return jsonify({"status": "error", "message": "Another exercise is running."})

    current_exercise = exercise_type
    exercise_thread = threading.Thread(target=run_exercise, args=(exercise_type,), daemon=True)
    exercise_thread.start()

    return jsonify({"status": "success", "message": f"{exercise_type} started."})

@app.route("/status")
def status():
    global exercise_module_obj

    if exercise_module_obj:
        with lock:
            reps = getattr(exercise_module_obj, "rep_count", 0)
            feedback = getattr(exercise_module_obj, "feedback", "Get into position.")
            done = getattr(exercise_module_obj, "done", False)
    else:
        reps = 0
        feedback = "No exercise running."
        done = False

    return jsonify({
        "reps": reps,
        "feedback": feedback,
        "done": done
    })

@app.route("/stop", methods=["POST"])
def stop_exercise():
    global exercise_thread, exercise_module_obj
    if exercise_thread and exercise_thread.is_alive():
        with lock:
            if exercise_module_obj:
                exercise_module_obj.done = True
        return jsonify({"status": "success", "message": "Exercise stopping soon."})
    return jsonify({"status": "error", "message": "No exercise running."})

# ================== MAIN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
