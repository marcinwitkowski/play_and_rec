"""
Code under MIT Licence,
by Marcin Witkowski, AGH, 2018
dsp.agh.edu.pl
"""

import sounddevice as sd
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np
import time
import os


def play_and_rec(pb_wav_fname, rec_wav_fname, normalise=True, rm_latency=True, show_plots=False):
    """
    Function plays and records simultaneously an audio file and stores the result in the same format as an input file.
        Output_dir is created if it does not exist just before writing the output file.
    :param pb_wav_fname:    input wavefile path
    :param rec_wav_fname:   output wavefile path
    :param normalise:       (default True) applies amplitude normalisation to the input samples
    :param rm_latency:      (default True) removes latency introduced during recording based on simple lookup for
    :param show_plots:      (default False) plots the Original(playback) and Recorded signals
    :return:                tuple with original samples, recorded samples and sampling rate
    """
    (fs, pb_samples) = read(pb_wav_fname)
    if normalise:
        pb_samples = pb_samples/(0.8 * pb_samples.max())  # normalise input
    pb_samples_padded = np.pad(pb_samples, (0, round(0.5*fs)), mode='constant', constant_values=0)
    rec_samples = sd.playrec(pb_samples_padded, fs, channels=1, blocking=True)
    if rm_latency:
        eps = 3 * 1.0/(2**15)      # assumes 16-bit WAV resolution
        ind = np.argmax(abs(rec_samples) > eps)
        rec_samples = rec_samples[ind:]  # removes latency
    output_path = os.path.dirname(rec_wav_fname)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    write(rec_wav_fname, fs, rec_samples)

    if show_plots:
        plot_signal(pb_samples, fs, 'Original', 1)
        plot_signal(rec_samples, fs, 'Recorded', 2)
    return rec_samples, pb_samples, fs


def plot_signal(x, fs, title, fig_num=1):
    plt.figure(fig_num)
    plt.title(title)
    t = np.linspace(0, len(x) / fs, len(x))
    plt.ylim([-1.2, 1.2])
    plt.grid()
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.plot(t, x)
    plt.show()


def get_wavfiles(input_dir, output_dir):
    """
    Function loads names and paths of all wavfiles in the directory and its subdirectories and returns list of tuples
    for furhter processing. Output_dir is created if it does not exist. Function copies the folder structure from input 
    to the output directory.
    :param input_dir:     Directory from which all wavfile paths are read 
    :param output_dir:    Directory where all files will be stored
    :return:              list of tuples (playback_wavpath, recorded_wavpath) 
    """
    wavfile_pairs = list()
    for root, directories, filenames in os.walk(input_dir):
        for filename in [f for f in filenames if (f.endswith(".WAV") or f.endswith(".wav"))]:
            in_file = os.path.join(root, filename)
            output_path = root.replace(input_dir, output_dir)
            out_file = os.path.join(output_path, filename)
            item = (in_file, out_file)
            wavfile_pairs.append(item)
    return wavfile_pairs


if __name__ == '__main__':
    input_dir = 'input_db_small'
    output_dir = 'output_db_small'
    wavfiles = get_wavfiles(input_dir, output_dir)
    for (pb, rec) in wavfiles:
        time.sleep(0.5)
        play_and_rec(pb, rec, show_plots=True)
