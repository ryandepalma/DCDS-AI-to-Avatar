#Laura Ozoria
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
import csv
import os

# paths
video_path = "30vids/VAN-EDU_SCOUN_0006_B_1_1_SOLO.mp4" 
model_path = "C:/Users/laura/Documents/DSSF/DCDS-AI-to-Avatar/pose_landmarker_full.task"
csv_path   = f"C:/Users/laura/Documents/DSSF/DCDS-AI-to-Avatar/{os.path.splitext(os.path.basename(video_path))[0]}_landmarks.csv"

# pose detector
pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO)
)

# connections
CONNECTIONS      = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]
HEAD_CONNECTIONS = [(0,7),(0,8),(7,11),(8,12),(9,10),(0,9),(0,10)]

# video
vid_capture = cv2.VideoCapture(video_path)
if not vid_capture.isOpened():
    print(f"ERROR: Could not open {video_path}"); exit()

# csv
landmark_names = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
    "LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT",
    "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY",
    "LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB",
    "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
    "LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL","RIGHT_HEEL",
    "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp_ms'] + [f"{n}_{a}" for n in landmark_names for a in ['x','y','z']])

    while vid_capture.isOpened():
        frame_read, curr_frame = vid_capture.read()
        if not frame_read: break

        h, w = curr_frame.shape[:2]
        timestamp_ms = int(vid_capture.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
        results = pose_detector.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            lm = results.pose_landmarks[0] # landmarks

            # save to csv
            row = [timestamp_ms]
            for l in lm:
                row.extend([l.x, l.y, l.z] if l.visibility > 0.5 else [None, None, None]) # body parts not visible in video
            writer.writerow(row)

            # draw dots and lines if visible in video
            for l in lm:
                if l.visibility > 0.5:
                    cv2.circle(curr_frame, (int(l.x*w), int(l.y*h)), 5, (0,255,0), -1)

            for s, e in CONNECTIONS + HEAD_CONNECTIONS:
                if lm[s].visibility > 0.5 and lm[e].visibility > 0.5:
                    color = (255,0,0) if (s,e) in HEAD_CONNECTIONS else (0,0,255)
                    cv2.line(curr_frame, (int(lm[s].x*w), int(lm[s].y*h)), (int(lm[e].x*w), int(lm[e].y*h)), color, 2)

        cv2.imshow("Pose Detection", curr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

vid_capture.release()
cv2.destroyAllWindows()
pose_detector.close()
print(f"Saved to: {csv_path}")