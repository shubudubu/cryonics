import os
import json
import librosa
import numpy as np
DATASET_PATH = "../dataset/"
OUTPUT_FILE = "baby_cry_features.json"
CRY_LABELS = {"belly_pain": 0, "burping": 1, "discomfort": 2, "hungry": 3, "tired": 4}
def extract_labels(file_path):
    return CRY_LABELS.get(os.path.basename(os.path.dirname(file_path)), -1)
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = {
            "Amplitude_Envelope_Mean": float(np.mean(np.abs(y))),
            "RMS_Mean": float(np.mean(librosa.feature.rms(y=y))),
            "ZCR_Mean": float(np.mean(librosa.feature.zero_crossing_rate(y))),
            "STFT_Mean": float(np.mean(np.abs(librosa.stft(y)))),
            "SC_Mean": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            "SBAN_Mean": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        }
        safe_n_bands = min(6, sr // 2000)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=safe_n_bands)
        features["SCON_Mean"] = float(np.mean(spectral_contrast))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"MFCCs{i+1}"] = float(np.mean(mfccs[i]))
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f"delMFCCs{i+1}"] = float(np.mean(delta_mfccs[i]))
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        for i in range(13):
            features[f"del2MFCCs{i+1}"] = float(np.mean(delta2_mfccs[i]))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        features["MelSpec"] = float(np.mean(mel_spec))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
def extract_features_and_save(dataset_path, output_file):
    data_list = []
    for cry_type, label in CRY_LABELS.items():
        folder_path = os.path.join(dataset_path, cry_type)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {cry_type} not found, skipping...")
            continue
        for file_name in filter(lambda f: f.endswith(".wav"), os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            features = preprocess_audio(file_path)
            if features:
                features.update({"Cry_Audio_File": file_name, "Cry_Reason": cry_type, "Label": label})
                data_list.append(features)
    with open(output_file, "w") as f:
        json.dump(data_list, f, indent=4)
    print(f"Feature extraction complete. Saved to {output_file}")
extract_features_and_save(DATASET_PATH, OUTPUT_FILE)
