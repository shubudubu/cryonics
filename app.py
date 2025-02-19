import json
import numpy as np
import pandas as pd
import joblib
import librosa
import os
from flask import Flask, request, jsonify, render_template

from flask import Flask, Response, render_template, jsonify, request, send_file
import cv2
import mediapipe as mp
import time
import threading

questions = {
    1: "Does your child enjoy being swung, bounced on your knee, etc.?",
    2: "Does your child take an interest in other children?",  # Critical
    3: "Does your child like climbing on things, such as up stairs?",
    4: "Does your child enjoy playing peek-a-boo/hide-and-seek?",
    5: "Does your child ever pretend, for example, to talk on the phone or take care of a doll?",
    6: "Does your child ever use his/her index finger to point, to ask for something?",
    7: "Does your child ever use his/her index finger to point, to indicate interest in something?",  # Critical
    8: "Can your child play properly with small toys (e.g. cars or blocks) without just mouthing, fiddling, or dropping them?",
    9: "Does your child ever bring objects over to you (parent) to show you something?",  # Critical
    10: "Does your child look you in the eye for more than a second or two?",
    11: "Does your child ever seem oversensitive to noise? (e.g., plugging ears)? (REVERSE)",
    12: "Does your child smile in response to your face or your smile?",
    13: "Does your child imitate you? (e.g., you make a face—will your child imitate it?)",  # Critical
    14: "Does your child respond to his/her name when you call?",  # Critical
    15: "If you point at a toy across the room, does your child look at it?",  # Critical
    16: "Does your child walk?",
    17: "Does your child look at things you are looking at?",
    18: "Does your child make unusual finger movements near his/her face? (REVERSE)",
    19: "Does your child try to attract your attention to his/her own activity?",
    20: "Have you ever wondered if your child is deaf? (REVERSE)",
    21: "Does your child understand what people say?",
    22: "Does your child sometimes stare at nothing or wander with no purpose? (REVERSE)",
    23: "Does your child look at your face to check your reaction when faced with something unfamiliar?"
}

# Critical questions
critical_items = {2, 7, 9, 13, 14, 15}

# Reverse scored items
reverse_items = {11, 18, 20, 22}

# Load models and scalers
model = joblib.load("baby_cry_classifier.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Load feature names from JSON
with open("baby_cry_features.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.drop(columns=["Cry_Audio_File", "Cry_Reason"], inplace=True)
feature_names = df.drop(columns=["Label"]).columns.tolist()

app = Flask(__name__)

# Feature Extraction Function
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        
        features = {
            "Amplitude_Envelope_Mean": np.mean(np.abs(y)),
            "RMS_Mean": np.mean(librosa.feature.rms(y=y)),
            "ZCR_Mean": np.mean(librosa.feature.zero_crossing_rate(y=y)),
            "STFT_Mean": np.mean(np.abs(librosa.stft(y))),
            "SC_Mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "SBAN_Mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "SCON_Mean": np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50.0, n_bands=4)),
            "MelSpec": np.mean(librosa.feature.melspectrogram(y=y, sr=sr))
        }

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"MFCCs{i+1}"] = np.mean(mfccs[i])

        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f"delMFCCs{i+1}"] = np.mean(delta_mfccs[i])

        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        for i in range(13):
            features[f"del2MFCCs{i+1}"] = np.mean(delta2_mfccs[i])

        return features
    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return None


# Prediction Function
def predict_baby_cry(new_cry_features):
    try:
        new_data = pd.DataFrame([new_cry_features], columns=feature_names)

        # Normalize features
        new_data_scaled = scaler.transform(new_data)

        # Select features
        new_data_selected = selector.transform(new_data_scaled)

        # Predict
        prediction = model.predict(new_data_selected)

        # Map prediction to cry type
        cry_types = {
            0: "Belly Pain",
            1: "Burping",
            2: "Discomfort",
            3: "Hungry",
            4: "Tired"
        }
        predicted_label = prediction[0]
        predicted_cry_type = cry_types.get(predicted_label, "Unknown")

        return predicted_cry_type
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error in Prediction"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Global variables for gaze tracking
right_gaze_frames = 0
total_frames = 0
is_tracking = False
start_time = None
capture_thread = None


def get_gaze_direction(landmarks):
    """Determine face direction based on facial landmarks."""
    left_eye = landmarks[33]  # Left eye outer corner
    right_eye = landmarks[263]  # Right eye outer corner
    nose = landmarks[1]  # Nose tip

    eye_center_x = (left_eye.x + right_eye.x) / 2

    if nose.x < eye_center_x - 0.02:  # Looking Right
        return "Looking Right"
    elif nose.x > eye_center_x + 0.02:  # Looking Left"
        return "Looking Left"
    else:
        return "Looking Center"


def process_gaze():
    """Process gaze tracking in a separate thread."""
    global right_gaze_frames, total_frames, is_tracking

    cap = cv2.VideoCapture(0)

    while is_tracking:
        ret, frame = cap.read()
        if not ret:
            continue

        total_frames += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = {i: lm for i, lm in enumerate(face_landmarks.landmark)}
                direction = get_gaze_direction(landmarks)

                if direction == "Looking Right":
                    right_gaze_frames += 1

        time.sleep(0.03)  # Reduce CPU usage

    cap.release()
# Flask Routes

@app.route("/")  # Home page
def main_page():
    return render_template("main_page.html")

@app.route("/subpage")
def sub_page():
    return render_template("subpage.html")

@app.route("/cu")
def cu():
    return render_template("cu.html")

@app.route("/detcry")
def detcry():
    return render_template("detcry.html")

@app.route("/at")
def at():
    return render_template("at.html")

@app.route("/agt")
def agt():
    return render_template("agt.html")

@app.route("/asc")
def asc():
    return render_template("asc.html",questions=questions)

@app.route("/results", methods=["POST"])
def results():
    responses = request.form
    total_score = 0
    critical_score = 0

    for num, response in responses.items():
        num = int(num)
        is_failed = (response == "no" and num not in reverse_items) or (response == "yes" and num in reverse_items)

        if is_failed:
            total_score += 1
            if num in critical_items:
                critical_score += 1

    # Determine risk level
    result_message = f"Total Score: {total_score} | Critical Score: {critical_score}<br>"

    follow_up_needed = False

    if total_score >= 10:
        result_message += "<b style='color: red;'>🚨 High Risk: Immediate referral to a specialist is needed.</b>"
    elif total_score >= 5:
        follow_up_needed = True
        result_message += "<b style='color: orange;'>⚠ Moderate Risk: Follow-Up Interview required.</b>"
        if critical_score >= 2 or total_score >= 3:
            result_message += "<br>🔎 Follow-Up suggests increased ASD risk. Referral recommended."
        else:
            result_message += "<br>✅ Follow-Up suggests no immediate concern."
    else:
        result_message += "<b style='color: green;'>✅ Low Risk: No further action needed.</b>"

    return render_template("result.html", result=result_message, follow_up_needed=follow_up_needed)

#cry det paths
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No file part"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file temporarily
    temp_path = "temp_audio.wav"
    audio_file.save(temp_path)

    # Extract features
    extracted_features = extract_features(temp_path)
    
    if extracted_features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    # Predict the crying type
    predicted_cry_type = predict_baby_cry(extracted_features)

    # Remove temp file
    os.remove(temp_path)

    return jsonify({"Crying Type": predicted_cry_type})

@app.route('/video_feed')
def video_feed():
    """Return the video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_test', methods=['POST'])
def start_test():
    """Start gaze tracking."""
    global right_gaze_frames, total_frames, is_tracking, capture_thread, start_time

    if is_tracking:
        return jsonify({"message": "Test already running!"})

    # Reset counters
    right_gaze_frames = 0
    total_frames = 0
    is_tracking = True
    start_time = time.time()

    # Start gaze processing in a separate thread
    capture_thread = threading.Thread(target=process_gaze)
    capture_thread.start()

    return jsonify({"message": "Test started!"})

#geometric paths
@app.route('/stop_test', methods=['POST'])
def stop_test():
    """Stop gaze tracking and return results."""
    global is_tracking

    if not is_tracking:
        return jsonify({"message": "Test not running!"})

    is_tracking = False  # Stop tracking

    if total_frames == 0:
        return jsonify({"error": "No data recorded"})

    right_gaze_percentage = (right_gaze_frames / total_frames) * 100
    result = {
        "right_gaze_percentage": round(right_gaze_percentage, 2),
        "autism_indicator": "Possible Autism Indicator" if right_gaze_percentage > 70 else "Normal"
    }

    return jsonify(result)


@app.route('/left_video')
def left_video():
    """Serve the left side video."""
    return send_file("static/left.mp4", mimetype="video/mp4")


@app.route('/right_video')
def right_video():
    """Serve the right side video."""
    return send_file("static/right.mp4", mimetype="video/mp4")



if __name__ == "__main__":
    app.run(debug=True)










