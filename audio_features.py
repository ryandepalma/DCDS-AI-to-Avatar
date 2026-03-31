import librosa
import numpy as np
import pandas as pd


file = "/Users/zixizeng/Desktop/FOR-MST_0023_0_2_1_SOLO.wav"


y, sr = librosa.load(file)


segment_length = 1  
samples_per_segment = segment_length * sr

results = []


for i in range(0, len(y), samples_per_segment):
    segment = y[i:i + samples_per_segment]

    if len(segment) < sr:
        continue

    start_time = i / sr
    end_time = (i + samples_per_segment) / sr

    energy = np.mean(librosa.feature.rms(y=segment))

    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch_values) > 0:
        avg_pitch = np.mean(pitch_values)
    else:
        avg_pitch = 0

    if energy > 0.02 and avg_pitch > 150:
        mood = "Engaged"
    elif energy < 0.01:
        mood = "Calm"
    else:
        mood = "Moderate"

    results.append({
        "Start_Time": start_time,
        "End_Time": end_time,
        "Energy": energy,
        "Pitch": avg_pitch,
        "Mood": mood
    })

df = pd.DataFrame(results)

df.to_excel("/Users/zixizeng/Desktop/audio_time_analysis.xlsx", index=False)

print("yes!Excel")
