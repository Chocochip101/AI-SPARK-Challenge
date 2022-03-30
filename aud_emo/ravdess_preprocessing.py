# author: Jeiyoon
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys

ravdess_path = "/content/drive/My Drive/Audio_Speech_Actors_01-24" # 1440 wav files
ravdess_file_path = []
ravdess_data = []
emotion = []
gender = []
actor = []

"""
{1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
name: 03-01-01-01-01-02-06.wav

path:  /content/drive/My Drive/Audio_Speech_Actors_01-24/Actor_06/03-01-01-01-01-02-06.wav
part:  ['/content/drive/My Drive/Audio_Speech_Actors_01', '24/Actor_06/03', '01', '01', '01', '01', '02', '06']
emotion:  [1]
actor:  [2]
bg:  2
"""
for path in tqdm(Path(ravdess_path).glob("**/*.wav")):
  emotion_dict = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
  part = str(path).split('.')[0].split('-')
  emotion.append(emotion_dict[int(part[2])])
  actor.append(int(part[6]))
  bg = int(part[6])
  
  if bg % 2 == 0:
    bg = "female"
  else:
    bg = "male"

  gender.append(bg)
  ravdess_file_path.append(path)
  
  name = str(path).split('/')[6].replace(".wav", "")
  
  try:
    # There are some broken files
    s = torchaudio.load(path)
    ravdess_data.append({
        "name": name,
        "path": path,
        "emotion": emotion
    })
  except Exception as e:
    pass
  
df = pd.DataFrame(data)
print(df.head())
