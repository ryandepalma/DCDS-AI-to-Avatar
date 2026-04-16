# AI Avatar Multimodal Analysis
This project builds a multimodal pipeline to analyze video data by combining audio, text, and motion features.

## Overview
The goal of this project is to understand how speech, tone, and body movements relate to meaning and engagement. The pipeline extracts and aligns multiple types of data from video:
* Speech content (transcription)
* Audio features (volume, pitch, energy)
* Body movement (pose landmarks)

## Pipeline
The project follows an end-to-end process:
1. Extract audio from video
2. Transcribe speech using Whisper
3. Analyze audio features (volume, pitch, energy)
4. Extract pose landmarks using MediaPipe
5. Align all data into time-based intervals
6. Generate structured output for further analysis

## Key Files
* `videoProcessor.py` → main pipeline combining all steps
* `audio transciptor.py` / `generateTranscript.py` → speech-to-text
* `audio_features.py` → energy, pitch, mood detection
* `decibel_calc.py` → volume and decibel changes
* `word_emphasis_analysis.py` → word-level sentiment and emphasis
* `MediaPipe_Test.py` → pose landmark extraction

## Methodology
* Whisper for speech recognition
* Librosa for audio feature extraction
* MediaPipe for pose detection
* Python (pandas, numpy) for data processing

Data is aggregated into time intervals (e.g., 5 seconds) to align text, audio, and motion features.

## Output
The pipeline generates:
* CSV / Excel files containing:
  * timestamps
  * transcript text
  * audio features (volume, pitch, energy)
  * word-level statistics
  * pose landmark data

## Applications
* AI avatar training
* Speech behavior analysis
* Emotion and engagement detection
* Human-computer interaction research

## Author
Ryan DePalma
Laura Ozoria Minaya
Zoey Zeng

## Pipeline Diagram
This diagram shows the full multimodal pipeline used in this project.
           ┌──────────────────────┐
           │      Input Video     │
           └─────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐       ┌────────────────┐
│ Extract Audio │       │ Extract Frames │
│ (moviepy)     │       │ (OpenCV)       │
└──────┬────────┘       └──────┬─────────┘
       │                       │
       ▼                       ▼
┌───────────────┐     ┌────────────────────┐
│ Transcription │     │ Pose Detection     │
│ (Whisper)     │     │ (MediaPipe)        │
└──────┬────────┘     └─────────┬──────────┘
       │                        │
       ▼                        ▼
┌────────────────┐     ┌────────────────────┐
│ Audio Features │     │ Landmark Data      │
│ (volume, pitch)│     │ (body movement)    │
│ (librosa)      │     └─────────┬──────────┘
└──────┬─────────┘               │
       │                         │
       └────────────┬────────────┘
                    ▼
      ┌──────────────────────────┐
      │ Combined Dataset Output  │
      │ (CSV / Excel)            │
      └────────────---───────────┘
```

