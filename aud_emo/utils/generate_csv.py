# author: Jeiyoon
from tqdm import tqdm
from pathlib import Path

import pandas as pd

tvsum_info_path = "/root/clip/ydata-tvsum50-info.tsv"

df_info = pd.read_csv(tvsum_info_path, sep='\t')
task_list = list(df_info['video_id'])

print("task_list: ", task_list)

# e.g. AwmHb44_ouw
# task_name = "98MoyGZKHXc"

normed_wav_cut_path = "/root/clip/video_datasets/normed_wav_cut"
normed_wav_save_path = "/root/clip/video_datasets/normed_wav_cut_csv"
normed_wav_cut_data = []

for task_name in task_list:
    for path in tqdm(Path(normed_wav_cut_path).glob("**/*.wav")):
        name = str(path).split('/')[5].replace('.wav', '')
        idx = str(path).split('_')[-1].replace('.wav', '')

        if task_name in name:
            try:
                  normed_wav_cut_data.append({
                      "index": int(idx),
                      "name": name,
                      "path": path,
                      "emotion": 'masked'
                  })
            except Exception as e:
                pass
        else:
            continue


    df_normed_wav_cut = pd.DataFrame(normed_wav_cut_data)
    print(df_normed_wav_cut)
    print(len(df_normed_wav_cut['name']))
    df_normed_wav_cut.to_csv(f"{normed_wav_save_path}/" + task_name + ".csv", sep="\t", encoding="utf-8", index=False)
