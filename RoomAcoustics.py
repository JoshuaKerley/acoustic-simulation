import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra

# room dimensions
rWidth = 4
rLength = 6
rHeight = 2

# barrier dimensions
bWidth = 2
bHeight = 1
bLocation = 3

# speaker location
speakerX = 2
speakerY = 1
speakerZ = 0.5

# microphone location
micX = 2
micY = 5
micZ = 0.5

# sound source file
source = "sweep-20Hz-20000Hz.wav"


# ---------------------------------------------------------
# create room
room = pra.ShoeBox([rWidth, rLength, rHeight])

# define barrier and add to room
bZeroX = (rWidth - bWidth) / 2
bCorners = [[bZeroX, bWidth + bZeroX, bWidth + bZeroX, bZeroX], [bLocation, bLocation, bLocation, bLocation], [0, 0, bHeight, bHeight]]
barrier = pra.Wall(bCorners, absorption=1, name="barrier")
room.walls.append(barrier)

# add speaker and attach sound source file
fs, signal = wavfile.read(source)
room.add_source([speakerX, speakerY, speakerZ], signal=signal)

# add microphone
room.add_microphone_array(pra.MicrophoneArray(np.array([[micX, micY, micZ]]).T, room.fs))

# visualize room
fig, ax = room.plot()
ax.set_xlim([-1, rWidth + 1])
ax.set_ylim([-1, rLength + 1])
ax.set_zlim([-1, rHeight + 1])
plt.show()