# 영상파일 오디오 변환, 저장
# for path in tqdm(Path(data_path).glob("**/*.wav")):
video_data_path = "/root/clip/video_datasets/SumMe/SumMe/videos/"
print(video_data_path + "Air_Force_One.mp4")
video_clip = mp.VideoFileClip(video_data_path + "Air_Force_One.mp4")
video_clip.audio.write_audiofile("/root/clip/video_datasets/SumMe/SumMe/video_to_audio/Air_Force_One.mp3")


# load mp3 file
# sr = 44100
y, sr = librosa.load("/root/clip/video_datasets/SumMe/SumMe/video_to_audio/Air_Force_One.mp3", sr = None)
# print(sr)

# mp3 to wav file
src = "/root/clip/video_datasets/SumMe/SumMe/video_to_audio/Air_Force_One.mp3"
dst = "/root/clip/video_datasets/SumMe/SumMe/video_to_audio/test.wav"

sound = AudioSegment.from_mp3(src) # load file as extention 'mp3'
sound.export(dst, format="wav") # convert file to wav

def trim_audio_data(audio_file, save_file):
  sr = 96000
  sec = 30

  y, sr = librosa.load(audio_file, sr = sr)

  ny = y[:sr*sec]

  sf.write(save_file+".wav", y, sr, format='WAV', endian='LITTLE', subtype='PCM_16')

#base_path = 'dataset/'
#audio_path = base_path + '/audio'

audio_path = "/root/clip/video_datasets/SumMe/SumMe/video_to_audio"
save_path = audio_path + '/save'

audio_list = os.listdir(audio_path)

for audio_name in audio_list:
  if audio_name.find('wav') is not -1:
      audio_file = audio_path + '/' + audio_name
      save_file = save_path + '/' + audio_name[:-4]

      trim_audio_data(audio_file, save_file)
