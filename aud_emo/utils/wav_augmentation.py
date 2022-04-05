# author: Jeiyoon
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torchaudio
import librosa
import matplotlib.pyplot as plt

from scipy.io import wavfile

os.environ["WANDB_DISABLED"] = "true"
aug_ravdess_path = "/root/clip/aug_ravdess"

ravdess_data = []
ravdess_path = "/root/clip/norm_ravdess"  # 1440 wav files
ravdess_file_path = []


def plot_time_series(data):
    fig = plt.figure(figsize=(10, 4))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()

def adding_white_noise(name, data) -> None:  # noise 방식으로 일반적으로 쓰는 잡음 끼게 하는 겁니다.
    sr = 44100
    noise_rate = 0.005

    wn = np.random.randn(len(data))
    data_wn = data + noise_rate * wn
    # plot_time_series(data_wn)

    data_wn = np.float32(data_wn)
    wavfile.write(aug_ravdess_path + "/" + "white_noise" + "/" + name + ".wav", sr, data_wn)

def shifting_sound(name, data) -> None:
    sr = 44100
    roll_rate = 0.1

    data_roll = np.roll(data, int(len(data) * roll_rate))
    # plot_time_series(data_roll)

    data_roll = np.float32(data_roll)
    wavfile.write(aug_ravdess_path + "/" + "shifting" + "/" + name + ".wav", sr, data_roll)

def stretch_sound(name, data) -> None:
    sr = 44100
    rate = 0.8

    data_stretch = librosa.effects.time_stretch(data, rate)
    # plot_time_series(data_stretch)

    data_stretch = np.float32(data_stretch)
    wavfile.write(aug_ravdess_path + "/" + "stretch" + "/" + name + ".wav", sr, data_stretch)

def reverse_sound(name, data) -> None:


    data_len = len(data)
    # data_reverse = np.array([data[len(data)-1-i] for i in range(len(data))])
    data_reverse = np.array([data[data_len - 1 - i] for i in range(data_len)])
    # plot_time_series(data_reverse)

    data_reverse = np.float32(data_reverse)
    wavfile.write(aug_ravdess_path + "/" + "reverse" + "/" + name + ".wav", sr, data_reverse)

def minus_sound(name, data) -> None:
    sr = 44100

    data_minus = (-1)*data
    # plot_time_series(data_minus)

    data_minus = np.float32(data_minus)
    wavfile.write(aug_ravdess_path + "/" + "minus" + "/" + name + ".wav", sr, data_minus)

    return data


for path in tqdm(Path(ravdess_path).glob("**/*.wav")):
    emotion_dict = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
    name = str(path).split('/')[5].replace(".wav", "")
    part = str(path).split('.')[0].split('-')
    label = str(emotion_dict[int(part[2])])

    if label == 'neutral':
      pass
    elif label == 'calm':
      label = 'neutral'
    else: # happy, sad, angry, fear, disgust, surprise
      label = 'emotion'

    try:
        # There are some broken files
        s = torchaudio.load(path)
        ravdess_data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        pass


df = pd.DataFrame(ravdess_data)
print(df.head())
print(df['emotion'].value_counts())


sr1, data1 = wavfile.read("/root/clip/norm_ravdess/Actor_05/03-01-02-01-02-01-05.wav")
# adding_white_noise("03-01-02-01-02-02-05", data)

sr2, data2 = wavfile.read("/root/clip/norm_ravdess/Actor_05/03-01-02-01-02-02-05.wav")

data3 = data2.reshape(-1)


for idx in tqdm(range(len(df['path']))):
    if df['emotion'][idx] == 'neutral':
        sr, data = wavfile.read(df['path'][idx])

        # shape error
        if len(data.shape) == 2:
            data = data.reshape(-1)

        # plot_time_series(data)
        # (1) white noise
        adding_white_noise(str(df['name'][idx]), data)
        # (2) Shifting
        shifting_sound(str(df['name'][idx]), data)
        # (3) Stretching
        stretch_sound(str(df['name'][idx]), data)
        # (4) Reverse
        reverse_sound(str(df['name'][idx]), data)
        # (5) Minus
        minus_sound(str(df['name'][idx]), data)
