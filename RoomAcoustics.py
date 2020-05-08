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

# barrier dimensions - adding a new wall didn't work consistently,
# so I had to switch it to a built-in wall that matches the rest of the room
bWidth = 2          # how far it extrudes from main wall
bLocation = 3       # y-location from front wall (lower = closer to speaker)

# wall absorbtion (0.0 - 1.0)
absorbtion = 0.5

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

# # helper functions
# def addWall(corner1, corner2, name = "barrier", absorption = 1.0):
#     bCorners = [[corner1[0], corner2[0], corner2[0], corner1[0]], [corner1[1], corner2[1], corner2[1], corner1[1]], [corner1[2], corner1[2], corner2[2], corner2[2]]]
#     barrier = pra.Wall(bCorners, absorption, name)
#     barrier2 = pra.Wall(bCorners, absorption, name+"2")

#     room.walls.append(barrier)
#     room.absorption = np.append(room.absorption, absorption)
#     print(room.corners)
#     room.normals = np.append(room.normals, [[0],[1],[0]], axis=1)
#     room.corners = np.append(room.corners, [[corner1[0]],[corner1[1]],[corner1[2]]], axis=1)
#     room.wall_names.append(name)
#     # room.walls.append(barrier)
#     # np.append(room.absorption, absorption)
#     # np.append(room.normals, [0,-1,0])
#     # np.append(room.corners, [corner1[0], corner2[0], corner2[0], corner1[0]])

#     print(room.corners)
#     print(room.wall_names)

# ---------------------------------------------------------


# read sound file
fs, signal_in = wavfile.read(source)

# create room
corners = np.array([[0,0], [0,bLocation], [bWidth, bLocation], [0,bLocation], [0,rLength], [rWidth,rLength], [rWidth,0]]).T  # [x,y]
room = pra.Room.from_corners(corners, fs=fs, absorption=absorbtion)
room.extrude(rHeight)

# room = pra.ShoeBox([rWidth, rLength, rHeight], fs=fs, absorption=1.0)

# define barrier and add to room - DIDN'T WORK RIGHT
# addWall([(rWidth-bWidth)/2,bLocation,0],[rWidth-(rWidth-bWidth)/2,bLocation,bHeight])

# add speaker and attach sound file
room.add_source([speakerX, speakerY, speakerZ], signal=signal_in)

# add microphone
room.add_microphone_array(pra.MicrophoneArray(np.array([[micX, micY, micZ]]).T, room.fs))

# plot room impulse response
room.image_source_model(use_libroom=True)
room.plot_rir()
fig = plt.gcf()
fig.set_size_inches(10, 5)

room.simulate()
# room.mic_array.record(signal, fs)
# room.mic_array.to_wav("test.wav")

# input frequency distribution
dft_input = np.fft.fft(signal_in)
freq_input = np.fft.fftfreq(n=signal_in.shape[0],d=1./fs)
# output frequency distribution
signal_out = room.mic_array.signals[0,:]
dft_output = np.fft.fft(signal_out)
freq_output = np.fft.fftfreq(n=signal_out.shape[0],d=1./fs)

print("Input signal length(samples): " + str(len(signal_in)))
print("Output signal length(samples): " + str(len(signal_out)))

fig, axs = plt.subplots(3, 1, constrained_layout=False)

axs[0].plot(np.arange(0,len(signal_out))/fs,signal_out/1000)
axs[0].set_xlabel('time')
axs[0].set_ylabel('Signal at Mic')
fig.suptitle('Frequency Distributions', fontsize=16)

axs[1].plot(freq_input,dft_input.real)
axs[1].set_xlim(-21000, 21000)
axs[1].set_xlabel('frequency (Hz)')
axs[1].set_ylabel('Source Freq Dist')

axs[2].plot(freq_output,dft_output.real/500000)
axs[2].set_xlim(-21000,21000)
axs[2].set_xlabel('frequency (Hz)')
axs[2].set_ylabel('Mic Freq Dist')

# visualize room
fig, ax = room.plot()
ax.set_xlim([-1, rWidth + 1])
ax.set_ylim([-1, rLength + 1])
ax.set_zlim([-1, rHeight + 1])
plt.show()