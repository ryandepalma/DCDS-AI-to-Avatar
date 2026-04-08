import numpy as np
import cv2
import os

JOINT_NAMES = [
    'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
    'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB'
]

ARM_CONNECTIONS = [
    (0, 1),  # shoulder -> elbow
    (1, 2),  # elbow -> wrist
    (2, 3),  # wrist -> pinky
    (2, 4),  # wrist -> index
    (2, 5),  # wrist -> thumb
]

W, H = 600, 600  # canvas size
FPS = 20     

def visualize_sequence(csv_path, video_writer):
    sequence = np.loadtxt(csv_path, delimiter=",")

    for frame_idx, frame in enumerate(sequence):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)  # black background for generated video

        # reshape into (6 joints, 3 coords)
        joints = frame.reshape(6, 3)

        # convert normalized coords to pixel positions
        pixel_coords = []
        for joint in joints:
            x = int(joint[0] * W)
            y = int(joint[1] * H)
            pixel_coords.append((x, y))

        # draw structure
        for start, end in ARM_CONNECTIONS:
            cv2.line(canvas, pixel_coords[start], pixel_coords[end], (0, 0, 255), 2)

        # draw joints and labels
        for i, (x, y) in enumerate(pixel_coords):
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(canvas, JOINT_NAMES[i], (x + 8, y), 0.4, (255, 255, 255), 1)

        # shows sequence and joints
        cv2.putText(canvas, f"{os.path.basename(csv_path)} | frame {frame_idx+1}/30",
                    (10, 20), 0.5, (200, 200, 200), 1)

        cv2.imshow("generated arm movement", canvas)
        video_writer.write(canvas)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # write 10 blank frames between sequences
    for _ in range(10):
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        video_writer.write(blank)

# video writer
os.makedirs("generated", exist_ok=True)
output_video_path = "generated/generated_movement.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 format
video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (W, H))

# loop through generated sequences
generated_folder = "generated"
generated_files = sorted([f for f in os.listdir(generated_folder) if f.startswith("generated_sequence_")])

for f in generated_files:
    visualize_sequence(os.path.join(generated_folder, f), video_writer)

video_writer.release()
cv2.destroyAllWindows()