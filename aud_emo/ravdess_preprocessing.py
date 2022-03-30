# author: Jeiyoon
ravdess_path = "/content/drive/My Drive/Audio_Speech_Actors_01-24" # 1440 wav files
ravdess_file_path = []
ravdess_data = []
# emotion = []
gender = []
actor = []

"""
Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

Filename identifiers

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).

Vocal channel (01 = speech, 02 = song).

Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.

Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").

Repetition (01 = 1st repetition, 02 = 2nd repetition).

Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

e.g.) 
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
  # emotion.append(emotion_dict[int(part[2])])
  emotion = emotion_dict[int(part[3])]
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


df = pd.DataFrame(ravdess_data)
print(df.head())
