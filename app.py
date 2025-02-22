import datetime
import json
import numpy as np
import pandas as pd
import joblib
import librosa
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response, send_file
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import mediapipe as mp
import time
import threading
import requests
import random
import logging
from functools import wraps


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load questions
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
CORS(app)  # Allow frontend to talk to backend

# Configuration
app.secret_key = '2e26ff392480745ffc7a87937f472b68'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['cryoincsDB']
users_collection = db['users']
contacts_collection = db['contacts']  # Collection to store contact messages

@app.route('/contact', methods=['POST'])
def contact():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    # Extract form data
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    # Validate required fields
    if not name or not email or not message:
        return jsonify({"error": "All fields are required!"}), 400

    # Store the data in MongoDB
    contact_data = {
        'name': name,
        'email': email,
        'message': message,
        'timestamp': datetime.datetime.utcnow()  # Add a timestamp
    }
    contacts_collection.insert_one(contact_data)

    return jsonify({"message": "Message sent successfully!"}), 200



# Ensure indexes
users_collection.create_index('username', unique=True)

# Decorator to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    # Check for required fields
    required_fields = ['username', 'childName', 'email', 'password', 'dob', 'birthplace', 'weight', 'bloodGroup', 'gender', 'address']
    for field in required_fields:
        if field not in data or not data[field]:
            return jsonify({"error": f"{field} is required!"}), 400

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Check if username or email already exists
    if users_collection.find_one({'username': username}):
        return jsonify({"error": "Username already exists!"}), 400
    if users_collection.find_one({'email': email}):
        return jsonify({"error": "Email already registered!"}), 400

    # Hash the password
    hashed_password = generate_password_hash(password)

    # Insert the new user
    user_data = {
        'username': username,
        'childName': data.get('childName'),
        'email': email,
        'password': hashed_password,
        'dob': data.get('dob'),
        'birthplace': data.get('birthplace'),
        'weight': data.get('weight'),
        'bloodGroup': data.get('bloodGroup'),
        'gender': data.get('gender'),
        'address': data.get('address')
    }
    users_collection.insert_one(user_data)

    return jsonify({"message": "Registration successful!"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required!"}), 400

    user = users_collection.find_one({'email': email})
    if user and check_password_hash(user['password'], password):
        session['username'] = user['username']  # Store username in session
        return jsonify({"message": "Login successful!"}), 200
    else:
        return jsonify({"error": "Invalid email or password!"}), 401


@app.route('/dashboard')
@login_required
def dashboard():
    return f'Hello, {session["username"]}! Welcome to your dashboard.'

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

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
        logger.error(f"Feature Extraction Error: {e}")
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
        logger.error(f"Prediction Error: {e}")
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
@app.route("/")
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
    return render_template("asc.html", questions=questions)

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

# Cry detection paths
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

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

# Function to get chatbot response
def chatbot_response(user_input):
    prompt = f"You are a helpful chatbot that only answers questions about infant care, child care, and parenting. If a user asks something else, politely decline.\n\nUser: {user_input}\nBot:"
    
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

    if response.status_code == 200:
        full_text = response.json()[0]["generated_text"]

        # Extract only the bot's response
        if "Bot:" in full_text:
            bot_response = full_text.split("Bot:")[1].strip()
        else:
            bot_response = full_text.strip()
        
        # Convert numbered lists into bullet points
        bot_response = format_as_bullets(bot_response)
        
        return bot_response
    else:
        responses = [
        "Hmm, I'm not sure I can help with that, but if you've got any health-related questions, I'm here for you!",
        "That's a bit out of my zone, but I'm happy to chat about anything medical!",
        "I might not have the answer to that, but if you want to talk about your health, I'm all ears!",
        "Not sure I can help with that, but if there's something medical on your mind, I'm here to listen!",
        "That's a little outside my expertise, but if you need help with a health concern, just let me know!",
        "I'm here to help with medical-related questions! Let me know if you'd like to talk about symptoms, treatments, or health advice.",
        "I'm not sure I understand your question. If you're asking about a health concern, I'll do my best to help!",
        "That might be outside my expertise, but I can help with medical questions if you'd like!",
        "I'm designed to provide medical information. Let me know how I can help with your health-related queries!",
        "Hmm, I might not have the answer to that, but I'm here to support you with medical advice!"
        ]
        return random.choice(responses)

def format_as_bullets(text):
    """ Converts numbered lists (1., 2., etc.) into bullet points (-) """
    lines = text.split("\n")
    formatted_lines = []
    
    for line in lines:
        if line.strip().startswith(tuple(f"{i}." for i in range(1, 20))):  # Matches "1.", "2." etc.
            line = "- " + line.split(".", 1)[1].strip()
        formatted_lines.append(line)
    
    return "\n".join(formatted_lines)

# Serve frontend
@app.route("/talk")
def talk():
    return render_template("talk.html")

@app.route("/cpd")
def cpd():
    return render_template("cpd.html")

# API Endpoint for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Invalid request"}), 400  # Handle invalid input

    user_input = data["message"]
    bot_response = chatbot_response(user_input)
    return jsonify({"response": bot_response})

def load_ngos():
    file_path = "output.json"
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
            return data
        except json.JSONDecodeError:
            return []  # If JSON is invalid, return an empty list

@app.route("/sngo", methods=["GET", "POST"])
def index():
    ngos = []
    
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        city = request.form.get("city", "").strip().lower()

        if not name or not city:
            return render_template("sngo.html", error="Please enter both name and city!")

        all_ngos = load_ngos()

        # ✅ Fix: Ensure "District" is a string before processing
        ngos = [
            ngo for ngo in all_ngos 
            if isinstance(ngo.get("District"), str) and ngo["District"].strip().lower() == city
        ]

        return render_template("ngor.html", name=name, city=city.title(), ngos=ngos)

    return render_template("sngo.html")




# cp detetction
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Global Variables
cap = None
movement_data = []
running = False
results_output = {}
frame_global = None
frame_lock = threading.Lock()  # Lock for thread-safe access to frame_global

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def process_frame():
    global cap, movement_data, running, results_output
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Perform MediaPipe detection (in the background, not displayed)
            _, results = mediapipe_detection(frame, holistic)
            hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = {mp_pose.PoseLandmark(i).name: (lm.x, lm.y) for i, lm in enumerate(landmarks)}

                left_shoulder, right_shoulder = keypoints['LEFT_SHOULDER'], keypoints['RIGHT_SHOULDER']
                left_hip, right_hip = keypoints['LEFT_HIP'], keypoints['RIGHT_HIP']
                left_knee, right_knee = keypoints['LEFT_KNEE'], keypoints['RIGHT_KNEE']
                left_wrist, right_wrist = keypoints['LEFT_WRIST'], keypoints['RIGHT_WRIST']

                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                hip_diff = abs(left_hip[1] - right_hip[1])
                knee_diff = abs(left_knee[1] - right_knee[1])

                is_asymmetrical = shoulder_diff > 0.1 or hip_diff > 0.1
                is_scissoring = abs(left_hip[0] - right_hip[0]) < 0.05 and knee_diff > 0.15

                is_fisting = False
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                        finger_distances = [
                            abs(index_tip.x - middle_tip.x),
                            abs(middle_tip.x - ring_tip.x),
                            abs(ring_tip.x - pinky_tip.x),
                            abs(pinky_tip.x - thumb_tip.x)
                        ]

                        if all(dist < 0.03 for dist in finger_distances):
                            is_fisting = True

                movement_data.append({
                    'asymmetry': is_asymmetrical,
                    'scissoring': is_scissoring,
                    'fisting': is_fisting
                })

            # Update the global frame with the raw video feed (no landmarks)
            with frame_lock:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_global = buffer.tobytes()

            time.sleep(0.01)

    calculate_results()
    cap.release()
    hands.close()
    pose.close()

def calculate_results():
    global movement_data, results_output
    total_frames = len(movement_data)

    if total_frames == 0:
        results_output = {"error": "No frames captured."}
        return

    asymmetry_count = sum(d['asymmetry'] for d in movement_data)
    scissoring_count = sum(d['scissoring'] for d in movement_data)
    fisting_count = sum(d['fisting'] for d in movement_data)

    asymmetry_percentage = (asymmetry_count / total_frames) * 100
    scissoring_percentage = (scissoring_count / total_frames) * 100
    fisting_percentage = (fisting_count / total_frames) * 100

    risk_signs = sum([
        asymmetry_percentage > 70,
        scissoring_percentage > 20,
        fisting_percentage > 75
    ])

    if risk_signs >= 1:
        message = "Possible Signs of Cerebral Palsy Detected. Consult a doctor for further evaluation."
    else:
        message = "No major risk detected. Continue monitoring your child's development."

    results_output = {
        "asymmetry_percentage": f"{asymmetry_percentage:.2f}%",
        "scissoring_percentage": f"{scissoring_percentage:.2f}%",
        "fisting_percentage": f"{fisting_percentage:.2f}%",
        "message": message
    }

@app.route('/video_feed_for_cpd')
def video_feed_for_cpd():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_cpd', methods=['POST'])
def start_cpd():
    global cap, running, movement_data, results_output
    if not running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video device."})
        movement_data = []
        running = True
        threading.Thread(target=process_frame).start()
        return jsonify({"message": "Detection started."})
    else:
        return jsonify({"message": "Detection already running."})

@app.route('/stop_cpd', methods=['POST'])
def stop_cpd():
    global running, results_output
    running = False
    return jsonify(results_output)

def generate_frames():
    global frame_global, running, frame_lock
    while running:
        with frame_lock:
            if frame_global is not None:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_global + b'\r\n')
                except Exception as e:
                    print(f"Error generating frame: {e}")
                    break


if __name__ == "__main__":
    app.run(debug=True)
