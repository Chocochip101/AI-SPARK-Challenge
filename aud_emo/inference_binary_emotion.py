# author: Jeiyoon
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Tuple

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
"""
(1) normalize audio
"""
# wav_data = []
# unnorm_tvsum_path = "/root/clip/video_datasets/wav_cut"
# norm_tvsum_path = "/root/clip/video_datasets/normed_wav_cut"
#
# for path in tqdm(Path(unnorm_tvsum_path).glob("**/*.wav")):
#     name = str(path).split('/')[5]
#     samplerate, data = wavfile.read(path)
#     scaled_data = np.float32(np.array(data / np.max(np.abs(data))))
#
#     # wavfile.write(norm_ravdess_path + "/" + name, samplerate, scaled_data)
#     wavfile.write(norm_tvsum_path + "/" + name, 44100, scaled_data)

"""
(2) Generate CSV file
"""
# normed_wav_cut_path = "/root/clip/video_datasets/normed_wav_cut"
# normed_wav_cut_data = []
# normed_wav_save_path = "/root/clip/video_datasets/normed_wav_cut_csv"
#
# for path in tqdm(Path(normed_wav_cut_path).glob("**/*.wav")):
#     name = str(path).split('/')[5].replace('.wav', '')
#     idx = str(path).split('_')[-1].replace('.wav', '')
#
#     if "AwmHb44_ouw" in name:
#         try:
#               normed_wav_cut_data.append({
#                   "index": int(idx),
#                   "name": name,
#                   "path": path,
#                   "emotion": 'masked'
#               })
#         except Exception as e:
#             pass
#     else:
#         continue
#
#
# df_normed_wav_cut = pd.DataFrame(normed_wav_cut_data)
# print(df_normed_wav_cut)
# print(len(df_normed_wav_cut['name']))
# df_normed_wav_cut.to_csv(f"{normed_wav_save_path}/AwmHb44_ouw.csv", sep="\t", encoding="utf-8", index=False)

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

# def speech_file_to_array_fn(path, sampling_rate):
#     speech_array, _sampling_rate = torchaudio.load(path)
#     resampler = torchaudio.transforms.Resample(_sampling_rate)
#     speech = resampler(speech_array).squeeze().numpy()
#     return speech

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


# STYLES = """
# <style>
# div.display_data {
#     margin: 0 auto;
#     max-width: 500px;
# }
# table.xxx {
#     margin: 50px !important;
#     float: right !important;
#     clear: both !important;
# }
# table.xxx td {
#     min-width: 300px !important;
#     text-align: center !important;
# }
# </style>
# """.strip()

def prediction(df_row):
    path, emotion = df_row["path"], df_row["emotion"]
    df = pd.DataFrame([{"Emotion": emotion, "Sentence": "    "}])

    setup = {
        'border': 2,
        'show_dimensions': True,
        'justify': 'center',
        'classes': 'xxx',
        'escape': False,
    }
    # ipd.display(ipd.HTML(STYLES + df.to_html(**setup) + "<br />"))
    # speech, sr = torchaudio.load(path)
    # speech = speech[0].numpy().squeeze()
    # speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
    # ipd.display(ipd.Audio(data=np.asarray(speech), autoplay=True, rate=sampling_rate))

    outputs = predict(path, sampling_rate) # sr = 16000
    # r = pd.DataFrame(outputs)
    # ipd.display(ipd.HTML(STYLES + r.to_html(**setup) + "<br />"))

    # [{'Emotion': 'emotion', 'Score': '0.1%'}, {'Emotion': 'neutral', 'Score': '99.9%'}]
    # print("outputs: ", outputs)
    if float(outputs[0]['Score'].replace("%", "")) > float(outputs[1]['Score'].replace("%", "")): # emotion
        return 1
    else: # neutral
        return 0

    # return outputs

# test_rav = pd.read_csv("/root/clip/ravdess_save_path/validation.csv", sep="\t")
# test_rav_iloc = test_rav.iloc[0]
# print(prediction(test_rav_iloc))

test_tv = pd.read_csv("/root/clip/video_datasets/normed_wav_cut_csv/AwmHb44_ouw.csv", sep="\t")
# test_tv['name'] = sorted(test_tv['name'])
test_tv = test_tv.sort_values(by='index' ,ascending = True)

emotion_per_second = []

# print(prediction(test_tv.iloc[0]))

from collections import Counter
import matplotlib.pyplot as plt

for idx in tqdm(range(len(test_tv['name']))):
    # four seconds
    emotion_per_second.append(prediction(test_tv.iloc[idx]))
    emotion_per_second.append(prediction(test_tv.iloc[idx]))
    emotion_per_second.append(prediction(test_tv.iloc[idx]))
    emotion_per_second.append(prediction(test_tv.iloc[idx]))

print(Counter(emotion_per_second))

x = np.arange(0, len(emotion_per_second))
plt.plot(x, emotion_per_second, color = 'red')
plt.xlabel("Times (s)")
plt.ylabel("Emotion")
plt.show()
