"""
Laura Ozoria
Load landmark csvs, extracts right arm joints, normalizes, 
augments, and splits the data into sets for training and validation.

DON'T RUN THIS CODE -> RUN THROUGH TRAIN.PY
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

"""
UPDATE TO THE FOLDER/PATH THAT CONTAINS YOUR PROCESSED LANDMARK CSV
"""
landmarks_folder = "video_data"  
all_sequences = []

csv_files = [f for f in os.listdir(landmarks_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    print(f"processing: {csv_file}")
    pose_data = pd.read_csv(os.path.join(landmarks_folder, csv_file))

    # ONLY right arm joints -> can be change to different body parts
    right_arm_cols = [col for col in pose_data.columns if any(
        joint in col for joint in ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
                                    'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB'])]

    arm_data = pose_data[['timestamp_ms'] + right_arm_cols].dropna()

    # slides 30 frame movement clips
    values = arm_data[right_arm_cols].values
    for i in range(len(values) - 30):
        all_sequences.append(values[i : i + 30])

sequences = np.array(all_sequences)

# normalize video size to have comparable data through all of them 
scaler = MinMaxScaler()
seq_flat = sequences.reshape(-1, sequences.shape[2])  # flatten to normalize
seq_flat = scaler.fit_transform(seq_flat)            
sequences = seq_flat.reshape(sequences.shape) # reshape back

# save scaler so generate.py can convert output to real coordinates
joblib.dump(scaler, "scaler.pkl")
print("scaler saved.")

def augment_sequences(sequences):
    """
    Doubles the dataset by creating two variations of each sequence.
    - Noise: simulates variation between different people
    - Speed: simulates faster movements to increase data points
    
    NOT NECESSERY WITH LARGER DATASESTS
    """

    augmented = []
    for seq in sequences:
        # simulate natural variation between people
        noise = seq + np.random.normal(0, 0.01, seq.shape)
        # simulate faster and slower movements for diverse data
        indices = np.linspace(0, len(seq)-1, int(len(seq) * 0.8)).astype(int)
        speed_var = seq[indices]
        speed_var = np.array([speed_var[int(i * len(speed_var) / len(seq))] for i in range(len(seq))])

        augmented.append(noise)
        augmented.append(speed_var)
    return np.array(augmented)

sequences = np.concatenate([sequences, augment_sequences(sequences)])

# trainining split 
split = int(len(sequences) * 0.8)
train_sequences = sequences[:split]
val_sequences   = sequences[split:]