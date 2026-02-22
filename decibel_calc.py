"""
Laura Ozoria

Please download requirements: pip intall requirements.txt
"""

from pydub import AudioSegment
from pydub.utils import make_chunks
import csv

def calc_db_per_second(file_path, save_csv = False, csv_name= None):
    """
    Calculates change in decibels per second in 5 second 'chunks'
     Args:
        file_path (str): path to the mp3 file.
        save_to_csv (bool): set to True to save a csv file.
        csv_name (str): name of the new csv file.

    Returns:
        list: Decibels per second per n millisecond intervals.
    """

    audio = AudioSegment.from_file(file_path, format = "mp3")

    chunk_length = 5000 # in milliseconds (5 seconds)
    chunks = make_chunks(audio, chunk_length) # chunk_length intervals

    db_per_second = []
    prev_db_per_second = None

    mp3_name = file_path.split('/')[-1]

    csv_data = []

    print(f"Duration: {len(audio)/1000} seconds") # duration of the video

    for i, chunk in enumerate(chunks):

        # decibels relative to full scale (amplitude levels)
        current_db_per_second = chunk.dBFS # (0 = max ; -inf silence)

        # change in decibels
        if prev_db_per_second is not None:
            change = current_db_per_second - prev_db_per_second
            db_per_second.append(change)

            csv_data.append({
                'video_name': mp3_name,
                'chunk_number': i,
                'time_seconds': i * 5, # chunk number to seconds
                'db_level': round(current_db_per_second,2),
                'db_change': round(change,2)
            })

            print(f"Second {i}: {current_db_per_second:.2f} dBFS | {change:.2f} dB/s")
        else:
            # No change in first chunk
            csv_data.append({
                'video_name': mp3_name,
                'chunk_number': i,
                'time_seconds': i * 5,
                'db_level': round(current_db_per_second,2),
                'db_change': 0.0  # No change for first chunk
            })

            print(f"Second {i}: {current_db_per_second:.2f} dB/s")

        prev_db_per_second = current_db_per_second

        if save_csv:
            if '.' in mp3_name:
                base_name = mp3_name.split('.')[0]
            else:
                base_name = mp3_name
            csv_name = f"{base_name}_db_per_second.csv"

        with open(csv_name, 'w', newline='') as csvfile:
            field_names = ['video_name', 'chunk_number', 'time_seconds', 'db_level', 'db_change']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)

            writer.writeheader()
            writer.writerows(csv_data)

        print(f"\n Data Saved to {csv_name}")

    return db_per_second

if __name__ == "__main__":
    calc_db_per_second("/Users/laura/Documents/DSSF/audio/VAN-EDU_SCOUN_0006_10_5_3_SOLO.mp3",
    save_csv = True)









