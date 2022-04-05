# author: Jeiyoon
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile

import numpy as np

wav_data = []
unnorm_tvsum_path = "/root/clip/video_datasets/wav_cut"
norm_tvsum_path = "/root/clip/video_datasets/normed_wav_cut"

for path in tqdm(Path(unnorm_tvsum_path).glob("**/*.wav")):
    name = str(path).split('/')[5]
    samplerate, data = wavfile.read(path)
    scaled_data = np.float32(np.array(data / np.max(np.abs(data))))

    # wavfile.write(norm_ravdess_path + "/" + name, samplerate, scaled_data)
    wavfile.write(norm_tvsum_path + "/" + name, 44100, scaled_data)
