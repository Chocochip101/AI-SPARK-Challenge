# author: Jeiyoon
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt

file_path = "/content/drive/My Drive/a01 (1).wav"

samplerate, data = sio.wavfile.read(file_path)
print(samplerate)
print(data)
print(len(data))

times = np.arange(len(data))/float(samplerate)

plt.fill_between(times, data)
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()
