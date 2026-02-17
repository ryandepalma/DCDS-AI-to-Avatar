import whisper
import librosa
import numpy as np
import math
import csv


def generate_interval_transcript_csv(
    video_path,
    output_file,
    interval=5,
    model_name="base"
):
    """
    Generates a CSV with:
        - timestamp intervals
        - average volume (RMS)
        - words per interval
        - words per second
        - transcript text

    Uses Whisper word-level timestamps + librosa RMS audio analysis.
    """

    print("Loading Whisper model:", model_name)
    model = whisper.load_model(model_name)

    print("Transcribing video with word timestamps...")
    result = model.transcribe(video_path, word_timestamps=True)

    print("Loading audio for volume analysis...")
    audio, sr = librosa.load(video_path, sr=None)

    frame_length = 2048
    hop_length = 512

    # RMS volume values
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length,
        n_fft=frame_length
    )

    print("Processing intervals...")
    chunks = {}

    # Fill chunks with word-level timestamps
    for segment in result["segments"]:
        if "words" not in segment:
            continue

        for w in segment["words"]:
            w_start = w["start"]
            w_end = w["end"]
            word = w["word"].strip()

            # Determine the bucket
            bucket = math.floor(w_start / interval) * interval

            if bucket not in chunks:
                chunks[bucket] = {
                    "text": "",
                    "volumes": [],
                    "word_count": 0
                }

            # Add the word
            chunks[bucket]["text"] += " " + word
            chunks[bucket]["word_count"] += 1

            # Volume values for this word
            word_rms = rms[(times >= w_start) & (times <= w_end)]
            chunks[bucket]["volumes"].extend(word_rms)

    # Helper: timestamp formatting
    def format_timestamp(seconds):
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    print(f"Writing CSV → {output_file}")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "timestamp",
            "avg_volume",
            "words_per_interval",
            "words_per_second",
            "text"
        ])

        for bucket in sorted(chunks.keys()):
            avg_volume = (
                float(np.mean(chunks[bucket]["volumes"]))
                if chunks[bucket]["volumes"] else 0
            )

            word_count = chunks[bucket]["word_count"]
            words_per_second = word_count / interval

            start_str = format_timestamp(bucket)
            end_str = format_timestamp(bucket + interval)

            writer.writerow([
                f"{start_str} - {end_str}",
                f"{avg_volume:.4f}",
                word_count,
                f"{words_per_second:.3f}",
                chunks[bucket]["text"].strip()
            ])

    print("Done! CSV created successfully.")


# -----------------------------
# Usage:
# -----------------------------
generate_interval_transcript_csv(
    video_path="/Users/ryandepalma/Desktop/DCDS/vid/FOR-MST_0023_0_1_1_SOLO.mp4.mp4",
    output_file="output.csv",
    interval=5,
    model_name="base"
)
