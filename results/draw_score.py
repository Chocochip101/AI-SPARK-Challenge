# author: Jeiyoon
"""
paper: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf

TVSum50 dataset contains 50 videos collected from
YouTube using 10 categories from the TRECVid MED task as search queries:

- changing Vehicle Tire (VT),
- getting Vehicle Unstuck (VU)
- Grooming an Animal (GA),
- Making Sandwich (MS),
- ParKour (PK),
- PaRade (PR),
- Flash Mob gathering (FM),
- BeeKeeping (BK),
- attempting Bike Tricks (BT),
- Dog Show (DS).

the TVSum dataset provides human annotated importance scores for every two second of each video
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from tqdm import tqdm

wav_cut_path = "/root/clip/video_datasets/wav_cut"
tvsum_anno_path = "/root/clip/ydata-tvsum50-anno.tsv"
tvsum_info_path = "/root/clip/ydata-tvsum50-info.tsv"

df = pd.read_csv(tvsum_anno_path, sep='\t', names=["video_name", "task", "importance_score"])
# print(df)

# 'video_id', 'length'
df_info = pd.read_csv(tvsum_info_path, sep='\t')
video_info_dict = defaultdict()
# print(df_info['video_id'])
# print(df_info['length'])

for idx in range(len(df_info['video_id'])):
    video_id = df_info['video_id'][idx]
    video_length = df_info['length'][idx].split(":")
    video_length = int(video_length[0]) * 60 + int(video_length[1])

    video_info_dict[video_id] = video_length

print(video_info_dict)

# check fps: all the tvsum video fps are not equal
score_per_second = []
avg_importance_score = defaultdict()
video_counter = Counter(df['video_name'])
task_score_per_second = defaultdict()

for idx in tqdm(range(len(df['video_name']))): # 1000
    # print("idx: ", idx)
    # print("df['video_name'][idx]: ", df['video_name'][idx])
    # print("df['task'][idx]: ", df['task'][idx])

    importance_score = df['importance_score'][idx].split(',')

    task_name = str(df['video_name'][idx])


    for fps in range(1, len(importance_score) + 1):
        tvsum_fps = round((len(importance_score) / video_info_dict[str(df['video_name'][idx])]) * 2)

        if fps % tvsum_fps == 0:
            # the TVSum dataset provides human annotated importance scores for every two second of each video
            score_per_second.append(importance_score[fps])
            score_per_second.append(importance_score[fps])

    if task_name in task_score_per_second:
        task_score_per_second[task_name] += np.float32(np.array(score_per_second))
    else:
        task_score_per_second[task_name] = np.float32(np.array(score_per_second))

    score_per_second.clear()


# mean
for v_name in task_score_per_second: # e.g.) v_name = "AwmHb44_ouw"
    avg_importance_score[v_name] = task_score_per_second[v_name] / video_counter[v_name]

# visualization
for v_name in avg_importance_score:
    x = np.arange(0, avg_importance_score[v_name].shape[0])
    # x = np.arange(0, len_AwmHb44_ouw)

    plt.plot(x, avg_importance_score[v_name], color = 'limegreen')
    # plt.plot(x, avg_AwmHb44_ouw / 20, color = 'limegreen')

    plt.title(v_name, fontsize = 15)
    plt.xlabel("Times (s)")
    plt.ylabel("Importance Score")
    plt.rcParams["figure.figsize"] = (8, 1)
    plt.show()
