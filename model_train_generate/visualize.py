import numpy as np
import cv2
import os

# joint names in output order
JOINT_NAMES = [
    'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
    'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB'
]

# joint connections to draw arm skeleton
ARM_CONNECTIONS = [
    (0, 1),  # shoulder -> elbow
    (1, 2),  # elbow -> wrist
    (2, 3),  # wrist -> pinky
    (2, 4),  # wrist -> index
    (2, 5),  # wrist -> thumb
]

W, H   = 600, 600
FPS    = 20
MARGIN = 50  # keeps joints away from canvas edge

def visualize_sequence(csv_path, video_writer):
    sequence = np.loadtxt(csv_path, delimiter=",")  # shape: (30, 18)
    print(f"Visualizing: {csv_path}")

    # amplify movement by exaggerating difference from mean pose
    mean_pose = sequence.mean(axis=0)          # average position across all frames
    diff = sequence - mean_pose                # how far each frame is from average
    sequence = mean_pose + diff * 10           # exaggerate that difference by 10x

    # re-normalize x and y to fit canvas so movement is always visible
    xy_coords = sequence.reshape(150, 6, 3)[:, :, :2] # change first ## to amount of frames
    xy_min = xy_coords.min()
    xy_max = xy_coords.max()

    for frame_idx, frame in enumerate(sequence):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)  # black background

        # reshape into (6 joints, 3 coords)
        joints = frame.reshape(6, 3)

        # scale coords to canvas with margin
        pixel_coords = []
        for joint in joints:
            x = int((joint[0] - xy_min) / (xy_max - xy_min + 1e-8) * (W - 2*MARGIN) + MARGIN)
            y = int((joint[1] - xy_min) / (xy_max - xy_min + 1e-8) * (H - 2*MARGIN) + MARGIN)
            pixel_coords.append((x, y))

        # draw bones
        for start, end in ARM_CONNECTIONS:
            cv2.line(canvas, pixel_coords[start], pixel_coords[end], (0, 0, 255), 2)

        # draw joints and labels --> uncomment to show
        for i, (x, y) in enumerate(pixel_coords):
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(canvas, JOINT_NAMES[i], (x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # shows sequence and frame number
        cv2.putText(canvas, f"{os.path.basename(csv_path)} | Frame {frame_idx+1}/150",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Generated Arm Movement", canvas)
        video_writer.write(canvas)

        # press q to skip to next sequence
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # blank pause between sequences
    for _ in range(60): # 40 = 2 seconds
        video_writer.write(np.zeros((H, W, 3), dtype=np.uint8))

# video writer setup
os.makedirs("generated", exist_ok=True)
output_video_path = "generated/generated_movement.mp4"
fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (W, H))

# loop through all generated sequences
generated_folder = "generated"
generated_files  = sorted([f for f in os.listdir(generated_folder) if f.startswith("generated_sequence_")])
print(f"found {len(generated_files)} generated sequences")

for f in generated_files:
    visualize_sequence(os.path.join(generated_folder, f), video_writer)

video_writer.release()
cv2.destroyAllWindows()
print(f"video saved to: {output_video_path}")