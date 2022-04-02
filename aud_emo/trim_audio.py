import os
from pydub import AudioSegment
import moviepy.editor as mp
import math

# tvsum: "/root/clip/video_datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/"
# summe: "/root/clip/video_datasets/SumMe/SumMe/videos/"
video_dir = "/root/clip/video_datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/"

# tvsum: "/root/clip/video_datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video_to_audio/"
# summe: "/root/clip/video_datasets/SumMe/SumMe/video_to_audio/"
audio_dir = "/root/clip/video_datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video_to_audio/"
wave_cut_dir = "/root/clip/video_datasets/wav_cut/"

mp4_name_list = os.listdir(video_dir)
# print(mp4_name_list)

for mp4_name in mp4_name_list:
    if mp4_name.find('mp4') is not -1: # **.webm / **.mp4
        name = mp4_name[:-4]
        video_clip = mp.VideoFileClip(video_dir + name + ".mp4")

        # no audio signal
        if video_clip.audio == None:
            continue

        video_clip.audio.write_audiofile(audio_dir + name + ".mp3")

        src = audio_dir + name + ".mp3"
        dst = audio_dir + name + ".wav"
        sound = AudioSegment.from_mp3(src) # load file as extention 'mp3'
        sound.export(dst, format="wav") # convert file to wav

        sound2 = AudioSegment.from_mp3(dst)

        # two_seconds = 1 * 1000 # ms
        two_seconds = 1 * 1000 * 4 # ms

        # math.floor(): 실수를 내림하여 정수 반환
        # e.g.) len(sound2) = 169340
        """
        Disclaimer: for문에 길이 꼭 체크!!! (e.g. 1000 or 4000)
        """
        for i in range(int(math.floor(len(sound2) / 4000))): # 169
            slice = sound2[i * two_seconds : (i + 1) * two_seconds]
            slice.export(wave_cut_dir + name + '_{}.wav'.format(i), format="wav")
