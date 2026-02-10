
# help with downloading packages (run in terminal)

#=# whisper 
python3 -m pip install git+https://github.com/openai/whisper.git
# Or if you already have pip version 24+, you can use:
python3 -m pip install openai-whisper

#=# librosa
python3 -m pip install librosa

#=# numpy
python3 -m pip install numpy

#=# ffmpeg 
# *note: was more difficult for me due to homebrew
brew install ffmpeg
# home brew installer 
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# apple m1/m2 add to path
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
