from moviepy import VideoFileClip

video = VideoFileClip("/Users/zixizeng/Desktop/FOR-MST_0023_0_2_1_SOLO.mp4")

audio = video.audio
audio.write_audiofile("/Users/zixizeng/Desktop/FOR-MST_0023_0_2_1_SOLO.wav")

print("✅ Done!")