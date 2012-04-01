import wave
import struct

import numpy as np

class Sound():

    def __init__(self):
        self.sample_depth = 2 #in bytes
        self.sample_rate = 44100 #in Hz
        self.data = None
        self.num_channels = 1

    def to_wav(self, output_file):
        wf = wave.open(output_file, 'w')

        wf.setparams( (self.num_channels, self.sample_depth, self.sample_rate,
                       len(self.data), 'NONE', 'not compressed') )
        #normalize the sample
        nsound = ((self.data / np.abs(self.data).max())*32767.0).astype('int')
        hex_sound = [struct.pack('h', x) for x in nsound]
        wf.writeframes(''.join(hex_sound))
        wf.close()


def gen_sine_wave(dur, freq, samprate):

    t = np.arange(0.0, dur, 1.0 / samprate)
    return np.sin(2*np.pi*freq*t)



