import wave
import struct
import numpy as np
from scipy import signal
import pywt

# references:
# http://soundlab.cs.princeton.edu/publications/2001_amta_aadwt.pdf
# http://mziccard.me/2015/06/12/beats-detection-algorithms-2/
# https://github.com/scaperot/the-BPM-detector-python/blob/master/bpm_detection/bpm_detection.py

# data:
# http://drumslive.com/dir/free-loops/

def wavLoad(fname):
    wav = wave.open(fname, 'r')
    params = (nchannels, sampwidth, _, nframes, _, _) = wav.getparams()
    frames = wav.readframes(nframes * nchannels)
    fmt = "%dB" % (nframes * nchannels) if sampwidth == 1 else "%dH" % (nframes * nchannels)

    return (params, struct.unpack_from(fmt, frames))


def estimateBPM(fname):
    ((nchannels, _, framerate, total_frames, _, _), wav_data) = wavLoad(fname)
    print('# of channels {}'.format(nchannels))
    print('framerate {}'.format(framerate))

    left, right = [], []
    if nchannels == 2:
        left = wav_data[::2]
        right = wav_data[1::2]
    else:
        left = right = wav_data

    window_size = framerate * 3
    levels = 4

    bpms = []
    print('Computing left channel...')
    for i in xrange(window_size, len(left), window_size):
        bpm = computeWindowBPM(left[i - window_size: i], framerate, levels)
        print(bpm)
        bpms.append(bpm)

    print('Computing right channel...')
    for i in xrange(window_size, len(right), window_size):
        bpm = computeWindowBPM(right[i - window_size: i], framerate, levels)
        print(bpm)
        bpms.append(bpm)

    estimated_bpm = np.median(np.array(bpms))
    return estimated_bpm

def computeWindowBPM(data, framerate, levels):
    # 0) Extract DWTs
    dCs = extractDetailCoefficients(data, levels)

    # 0.5 ) Extract relevant variables
    max_decimation = 2**(levels - 1)
    dC_minlen = len(dCs[0])/max_decimation
    min_idx = 60. / 220 * (framerate/max_decimation)    # used to define bpm upper bound
    max_idx = 60. / 40 * (framerate/max_decimation)     # used to define bpm lower bound

    # 1) LPF
    dCs = [signal.lfilter([0.01], [1 -0.99], dC) for dC in dCs]

    # 2) FWR
    dCs = [[abs(datum) for datum in dC] for dC in dCs]

    # 3) DOWN
    # note the decimation order is reveresed in reference, look into this
    dCs = [dC[::2**i] for i, dC in enumerate(dCs)]

    # 4) NORM
    for i, dC in enumerate(dCs):
        mean = np.mean(dC)
        dCs[i] = [datum - mean for datum in dC]

    # 5) ACRL
    # note implementation details are vague, look into this
    # experiment with adding approximate data
    dC_sum = [0 for i in range(dC_minlen)]
    for i in reversed(range(levels)):
        dC_sum += dCs[i][:dC_minlen]                    # experiment with and without using minlen
    correl = np.correlate(dC_sum, dC_sum, 'full')

    # Extract peak
    correl_midpoint = len(correl)/2
    correl_half = correl[correl_midpoint:]
    peak_idx = detect_peak(correl_half[min_idx:max_idx])[0] + min_idx   # adjust for the data subset

    # Calculate BPM
    bpm = 60. / peak_idx * (framerate/max_decimation)   # do the math here
    return bpm


# simple peak detection
def detect_peak(data):
    max_val = np.amax(abs(data))
    peak_ndx = np.where(data==max_val)
    if len(peak_ndx[0]) == 0: #if nothing found then the max must be negative
        peak_ndx = np.where(data==-max_val)
    return peak_ndx


def extractDetailCoefficients(data, deg=4):
    dC_list = []
    for i in range(deg):
        aC, dC = pywt.dwt(data, 'db4')
        dC_list.append(dC)
        data = aC
    return dC_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-source', default='./data/daybreak.wav', help='wav file to be analyzed; defaults to ./data/daybreak.wav')
    args = parser.parse_args()

    print('Estimating BPM of {}'.format(args.source))
    estimated_bpm = estimateBPM(args.source)
    print('Estimated BPM: {}'.format(estimated_bpm))

