"""
Laura Ozoria

Please download requirements: pip intall requirements.txt
"""

from pydub import AudioSegment
from pydub.utils import make_chunks

## LIBROSA MAY BE USED TO ANALYZE FASE
# script_directory = os.path.dirname(os.path.abspath(__file__))
# file_name = "VAN-EDU_SCOUN_0006_10_5_3_SOLO.mp3"
#
# file_path = os.path.join(script_directory, file_name)

def calc_db_per_second(file_path):
    """
    Docstring
    Args:
        file_path (str): path to the mp3 file.

    Returns:
        list: Decibels per second per n millisecond intervals.
    """

    audio = AudioSegment.from_file(file_path, format = "mp3")

    chunk_length = 5000 # in milliseconds (5 seconds)
    chunks = make_chunks(audio, chunk_length) # chunk_length intervals

    db_per_second = []
    prev_db_per_second = None

    print(f"Duration: {len(audio)/1000} seconds") # duration of the video

    for i, chunk in enumerate(chunks):

        # decibels relative to full scale (amplitude levels)
        current_db_per_second = chunk.dBFS # (0 = max ; -inf silence)

        # change in decibels
        if prev_db_per_second is not None:
            change = current_db_per_second - prev_db_per_second
            db_per_second.append(change)

            print(f"Second {i}: {current_db_per_second:.2f} dBFS | {change:.2f} dB/s")
        else:
            print(f"Second {i}: {current_db_per_second:.2f} dB/s")

        prev_db_per_second = current_db_per_second

    return db_per_second

# write down path to mp3 file
#calc_db_per_second("/Users/laura/Documents/DSSF/audio/VAN-EDU_SCOUN_0006_10_5_3_SOLO.mp3")










