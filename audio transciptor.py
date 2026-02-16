import whisper
import librosa
import numpy as np
import math
import csv

# ===== USER CONFIG ===== #
video_path = "/Users/ryandepalma/Desktop/DCDS/vid/FOR-MST_0023_0_1_1_SOLO.mp4.mp4"   # replace with path
interval = 10       # set your custom interval in seconds
model_name = "base"                    # options: tiny, base, small, medium, large
output_file = "FOR-MST_0023_0_1_1_SOLO.csv"  # make sure to change file name for output
        # don't forget to include .txt ^
# ===== SUMMARY OF LOGIC ===== #
# 1. Whisper → get transcript with segments and timestamps
# 2. Librosa → get audio RMS (volume) per small frame
# 3. Map each Whisper segment into custom intervals
# 4. Aggregate:
#  - Combine text per interval
#  - Combine RMS values per interval
# 5. Output → file showing [interval timestamps] | average volume | transcript text
# ======================= #


# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model(model_name)

# Transcribe video
print("Transcribing video...")
result = model.transcribe(video_path)

# Load audio with librosa
print("Loading audio for volume analysis...")
audio, sr = librosa.load(video_path, sr=None) #sr keeps original sampling rate

# Compute RMS (root mean square of volume)
frame_length = 2048     # how many samples grouped per calculation
hop_length = 512       # how many samples to move forward to the next RMS value *
rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)

# Group segments into custom intervals
chunks = {}
# aligning segments to custom intervals
for segment in result["segments"]:  
    start = segment['start']
    end = segment['end']
    text = segment['text'].strip()
    
    start_bucket = math.floor(start / interval) * interval
    end_bucket = math.floor(end / interval) * interval
    
    # handling of segments that span intervals
    for bucket in range(int(start_bucket), int(end_bucket) + interval, interval):
        if bucket not in chunks:
            chunks[bucket] = {"text": "", "volumes": []}
        chunks[bucket]["text"] += " " + text
        segment_rms = rms[(times >= start) & (times <= end)]
        chunks[bucket]["volumes"].extend(segment_rms)

# ===== WRITE CSV OUTPUT (Single Timestamp Column) ===== #
print(f"Saving transcript with volume info to {output_file}...")

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"  # e.g., "01:23.45"

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    # header row
    writer.writerow(["timestamp", "avg_volume", "text"])
    
    for bucket in sorted(chunks.keys()):
        avg_volume = float(np.mean(chunks[bucket]["volumes"])) if chunks[bucket]["volumes"] else 0
        
        start_str = format_timestamp(bucket)
        end_str = format_timestamp(bucket + interval)

        timestamp = f"{start_str} - {end_str}"

        writer.writerow([
            timestamp,
            f"{avg_volume:.4f}",
            chunks[bucket]["text"].strip()
        ])

print("Done!")


