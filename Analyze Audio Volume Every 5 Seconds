[volume.py](https://github.com/user-attachments/files/25275231/volume.py)
import os
import numpy as np
import pandas as pd
import librosa
from moviepy import VideoFileClip
from faster_whisper import WhisperModel

VIDEO_PATH = "/Users/zixizeng/Desktop/FOR-MST_0023_0_1_1_SOLO.mp4.mp4"
AUDIO_PATH = "/Users/zixizeng/Desktop/audio.wav"
OUTPUT_CSV = "/Users/zixizeng/Desktop/video_analysis_5s.csv"

WINDOW = 5

print("Video exists:", os.path.exists(VIDEO_PATH))

print("Extracting audio from video...")
video = VideoFileClip(VIDEO_PATH)
video.audio.write_audiofile(AUDIO_PATH, logger=None)
video.close()



print("Computing volume...")
y, sr = librosa.load(AUDIO_PATH, sr=None)

hop_length = 512
rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
times = librosa.frames_to_time(
    np.arange(len(rms)), sr=sr, hop_length=hop_length
)

volume_df = pd.DataFrame({
    "time": times,
    "volume": rms
})


print("Running transcription...")
model = WhisperModel("base", compute_type="int8")

segments, _ = model.transcribe(AUDIO_PATH)

seg_rows = []
for seg in segments:
    seg_rows.append({
        "start": seg.start,
        "end": seg.end,
        "text": seg.text.strip()
    })

seg_df = pd.DataFrame(seg_rows)


print("Aggregating into 5-second windows...")

rows = []
t_min = volume_df["time"].min()
t_max = volume_df["time"].max()

current = t_min

while current < t_max:
    window_start = current
    window_end = current + WINDOW

    avg_volume = volume_df[
        (volume_df["time"] >= window_start) &
        (volume_df["time"] < window_end)
    ]["volume"].mean()

    texts = seg_df[
        (seg_df["end"] > window_start) &
        (seg_df["start"] < window_end)
    ]["text"]

    rows.append({
        "window_start": round(window_start, 2),
        "window_end": round(window_end, 2),
        "avg_volume": avg_volume,
        "text": " ".join(texts)
    })

    current += WINDOW

window_df = pd.DataFrame(rows)

# ==========================================


window_df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
print("\nPreview:")
print(window_df.head())
