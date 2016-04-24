import wave
import struct
import numpy as np
from scipy import signal
import pywt


"""
References:
Discrete wavelet transform (DWT) based beat detection paper: http://soundlab.cs.princeton.edu/publications/2001_amta_aadwt.pdf
DWT info: https://en.wikipedia.org/wiki/Discrete_wavelet_transform
Marco Ziccardi's Scala implementation: http://mziccard.me/2015/06/12/beats-detection-algorithms-2/
Scaperot's own Python implementation: https://github.com/scaperot/the-BPM-detector-python/blob/master/bpm_detection/bpm_detection.py
"""


def wavLoad(fname):
    """
    A quick function to load a wav file and extract some parameters. The only ones you really
    need to keep note of are the number of channels in the audio (nchannels), the framerate (framerate),
    and the actual data, which is unpacked by ```struct.unpack_from(fmt, frames)```. For more details
    about how this works, check out the Python wave module documentation: https://docs.python.org/2/library/wave.html
    """
    wav = wave.open(fname, 'r')
    params = (nchannels, sampwidth, framerate, nframes, _, _) = wav.getparams()
    frames = wav.readframes(nframes * nchannels)

    """
    .wav files can be encoded in different formats. The format is given by the sample width
    which in turn determines the format character we should use to actually unpack the data
    """
    fmt = "%dB" % (nframes * nchannels) if sampwidth == 1 else "%dH" % (nframes * nchannels)

    return (params, struct.unpack_from(fmt, frames))


def decomposeChannels(nchannels, wav_data):
    """
    A .wav file may have more than one channel of audio. The data from each channel is interleaved
    in the unpacked wav data. This function separates the data into however many channels there
    may be. For the purposes of beat detection, though, we really only need one channel.
    """
    channels = []
    for i in range(nchannels):
        data = wav_data[i::nchannels]
        channels.append(data)
    return channels


def estimateBPM(fname, window_length=3, levels=5):
    """
    Here is where we actually determine the estimated bpm for the entire song. For windows of
    length ```window_length``` in seconds, we estimate the bpm of that window using
    ```computeWindowBPM```. The bpm of the entire song is determined by the median of
    each window's bpm.
    """

    ((nchannels, _, framerate, total_frames, _, _), wav_data) = wavLoad(fname)  # go ahead and get the wav data

    left = decomposeChannels(nchannels, wav_data)[0]                            # we only need one channel of sound data
    window_size = framerate * window_length                                     # number of frames in a window ((n frames / 1 second) * (m seconds))

    bpms = [computeWindowBPM(left[(i - window_size):i], framerate, levels) \
                for i in range(window_size, len(left), window_size)]            # get the bpms for every window

    estimated_bpm = np.median(np.array(bpms))                                   # the median bpm is the bpm of the song
    return estimated_bpm


def extractDWTCoefficients(data, deg=4):
    """
    Before we get into computing the window bpm, let's figure out exactly what the DWT is.
    Those familiar with signal processing will know the Fourier Transform, which decomposes a
    signal into its constituents. On a high level, the DWT does a similar decomposition, but only
    into high-frequency and low-frequency components. These are stored in the
    detail coefficients (dC) and approximation coefficients (aC) respectively. By performing cascading
    DWTs on the approximation coefficients, we can recover finer frequency resolution, which is we want for a
    good frequency decomposition: https://en.wikipedia.org/wiki/Discrete_wavelet_transform#Cascading_and_filter_banks
    """
    dC_list, aC_list = [], []
    for i in range(deg):
        aC, dC = pywt.dwt(data, 'db4')                                          # We use the 4 coefficient Daubechies wavelets because the paper says so
        dC_list.append(dC)                                                      # The length of each cascading transform is approximately halved
        aC_list.append(aC)                                                      # Here's why: http://www.pybytes.com/pywavelets/ref/dwt-discrete-wavelet-transform.html#single-level-dwt
        data = aC
    return dC_list, aC_list


def detectPeak(data):
    """
    All we're doing here is determining the index of the largest magnitude datum
    in the supplied data. This is needed later for determining the index of beats
    in a window. (Full disclosure, I just copied this part verbatim from Scaperot)
    """
    max_val = np.amax(abs(data))
    peak_ndx = np.where(data==max_val)
    if len(peak_ndx[0]) == 0:
        peak_ndx = np.where(data==-max_val)
    return peak_ndx


def computeWindowBPM(data, framerate, levels):
    """
    Finally, the real meat of the process: computing the bpm of a sound window.
    I'll annotate step-by-step and try to correlate the steps with the paper
    linked above.
    """

    # 0) Extract DWTs
    dCs, aCs = extractDWTCoefficients(data, levels)                             # We're going to need the high frequency decomposition (dC) and the first low frequency decomposition (aC).

    # 0.5 ) Extract relevant variables
    max_downsample = 2**(levels - 1)                                            # This will be useful later for downsampling and calculating the final bpm
    coeff_minlen = len(dCs[0])/max_downsample                                   # We'll use this later to ensure the size of each window is the same during computation
    min_idx = 60. / 220 * (framerate/max_downsample)                            # Here we define the upper and lower bounds for tempos our program can find.
    max_idx = 60. / 40 * (framerate/max_downsample)                             # 220bpm = upper bound, 40bpm = lower bound

    # 1) Low Pass Filter (LPF)
    dCs = [signal.lfilter([0.01], [1 -0.99], dC) for dC in dCs]                 # A low pass filter cleans up the noise at each frequency band

    # 2) Full Wave Rectification (FWR)
    dCs = [abs(dC) for dC in dCs]                                               # We want to make sure our signal values are >= 0

    # 3) Downsampling (DOWN)
    dCs = [dC[::2**(levels - 1 - i)] for i, dC in enumerate(dCs)]               # Remember how the size of each frequency band is roughly half the previous? Downsampling roughly equivocates the length of the bands

    # 4) Normalization (NORM)
    for i, dC in enumerate(dCs):
        mean = np.mean(dC)
        dCs[i] = dC - mean                                                      # Normalize the data just by subtracting the mean

    # 5) Autocorrelation (ACRL)
    dC_sum = np.zeros([coeff_minlen])                                           # We're going to perform the autocorrelation on the summation of each band
    for dC in dCs:
        dC_sum += dC[:coeff_minlen]                                             # To sum the bands up, each length needs to be exactly the same. Here's where coeff_minlen is handy!
    correlation = np.correlate(dC_sum, dC_sum, 'full')                          # We're looking for pattern similarities at different times, which correspond to beats!

    correlation_half = correlation[len(correlation)/2:]                         # The autocorrelation is symmetric, so we only need the latter half
    peak_idx = detectPeak(correlation_half[min_idx:max_idx])[0] + min_idx       # We just want to get the index of the highest peak in the data, which we can use to find the bpm!

    bpm = 60. / peak_idx * (framerate/max_downsample)                           # Just like when we were determining the bounds, we can use the same formula to determine the bpm
    print(bpm)
    return bpm


if __name__ == '__main__':
    """
    That should be just about it! Let's try it out!
    One optimization that I've noticed in other implementations
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
        default='./data/gold_dust_short.wav',
        help='wav file to be analyzed (defaults to ./data/gold_dust_short.wav)')
    args = parser.parse_args()

    print('Estimating BPM of {}'.format(args.filename))
    estimated_bpm = estimateBPM(args.filename)
    print('Estimated BPM: {}'.format(estimated_bpm))
