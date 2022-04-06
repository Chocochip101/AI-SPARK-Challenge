# author: Jeiyoon
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torchaudio

from transformers.file_utils import ModelOutput
from transformers import AutoConfig, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name_or_path = "/content/drive/My Drive/wav2vec2-large-xlsr-53-english/checkpoint-280"
model_name_or_path = "/root/clip/wav2vec2-large-xlsr-53-english/checkpoint-280"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
sampling_rate = processor.feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    # if speech_array.size() == 2:
    #     speech_array.view([1, -1])

    # exception: shape
    if len(speech) == 2:
        speech = speech[1]

    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs


def prediction(df_row):
    path, emotion = df_row["path"], df_row["emotion"]

    outputs = predict(path, sampling_rate) # sr = 16000

    # [{'Emotion': 'emotion', 'Score': '0.1%'}, {'Emotion': 'neutral', 'Score': '99.9%'}]
    # print("outputs: ", outputs)
    if float(outputs[0]['Score'].replace("%", "")) > float(outputs[1]['Score'].replace("%", "")): # emotion
        return 5 # max importance score
    else: # neutral
        return 0

    # return outputs


test_tvsum_path = "/root/clip/video_datasets/normed_wav_cut_csv"
task_emotion_dict = defaultdict()

# e.g.) path: /root/clip/video_datasets/normed_wav_cut_csv/Bhxk-O1Y7Ho.csv
for path in Path(test_tvsum_path).glob("**/*.csv"): # 50
    test_tv = pd.read_csv(path, sep="\t")
    # test_tv['name'] = sorted(test_tv['name'])
    test_tv = test_tv.sort_values(by='index', ascending=True)

    emotion_per_second = []
    task_name = str(path).split('/')[5].replace(".csv", "")
    print("task_name: ", task_name)

    # {'AwmHb44_ouw': 354, '98MoyGZKHXc': 187, 'J0nA4VgnoCo': 584, 'gzDbaEs1Rlg': 288, 'XzYM3PfTM4w': 111, 'HT5vyqe0Xaw': 322, 'sTEELN-vY30': 149, 'vdmoEJ5YbrQ': 329, 'xwqBXPGE9pQ': 233, 'akI8YFjEmUw': 133, 'i3wAGJaaktw': 156, 'Bhxk-O1Y7Ho': 450, '0tmA_C6XwfM': 141, '3eYKfiOEJNs': 194, 'xxdtq8mxegs': 144, 'WG0MBPpPC6I': 397, 'Hl-__g2gn_A': 243, 'Yi4Ij2NM7U4': 405, '37rzWOQsNIw': 191, 'LRw_obCPUt0': 260, 'cjibtmSLxQ4': 647, 'b626MiF1ew4': 235, 'XkqCExn6_Us': 188, 'GsAD1KT1xo8': 145, 'PJrm840pAUI': 274, '91IHQYk1IQM': 110, 'RBCABdttQmI': 364, 'z_6gVvQb2d0': 276, 'fWutDQy1nnY': 585, '4wU_LUjG5Ic': 167, 'VuWGsYPqAX8': 216, 'JKpqYvAdIsw': 152, 'xmEERLqJ2kU': 446, 'byxOvuiIJV0': 154, '_xMr-HKMfVA': 149, 'WxtbjNsCQ8A': 265, 'uGu_10sucQo': 167, 'EE-bNr36nyA': 98, 'Se3oxnaPsz0': 138, 'oDXZc0tZe04': 380, 'qqR6AEXwxoQ': 269, 'EYqVtI9YWJA': 198, 'eQu1rNs0an0': 164, 'JgHubY5Vw3Y': 143, 'iVt07TCkFM0': 104, 'E11zDS9XGzg': 510, 'NyBmCxDoHJU': 189, 'kLxoNp-UchI': 130, 'jcoYJXDG9sw': 199, '-esJrBWj2d8': 230}
    # e.g.) 'Bhxk-O1Y7Ho': 450 / 7:30
    for idx in tqdm(range(len(test_tv['name']))):
        # four seconds / mean
        # emotion_per_second.append(prediction(test_tv.iloc[idx]))
        emotion_per_second.append(prediction(test_tv.iloc[idx]))
        emotion_per_second.append(0)
        emotion_per_second.append(prediction(test_tv.iloc[idx]))
        emotion_per_second.append(0)
        # emotion_per_second.append(prediction(test_tv.iloc[idx]))

    print(Counter(emotion_per_second))
    print("emotion_per_second: ", emotion_per_second)
    task_emotion_dict[task_name] = emotion_per_second

"""
importance score
"""
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

    plt.figure(figsize=(18, 6))
    plt.bar(x, task_emotion_dict[v_name], color='peachpuff', alpha=0.9)
    plt.plot(x, avg_importance_score[v_name], color = 'firebrick')
    plt.legend(['importance score', 'emotion'])

    plt.title(v_name, fontsize = 15)
    plt.xlabel("Times (s)")
    plt.ylabel("Importance Score")
    plt.show()
