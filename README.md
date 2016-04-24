# Estimate BPM
A quick script to estimate the tempo of a wav file in beats per minute (bpm). The code is fully annotated with the intention of making the algorithm as understandable as possible. More details on how it exactly works can be found in the references linked below. On a high level, the algorithm performs Discrete Wavelet Transforms on sound windows, and autocorrelates the result to find where the beats lay.

# Usage
```
usage: bpm_estimator.py [-h] --filename FILENAME

optional arguments:
  -h, --help           show this help message and exit
  --filename FILENAME  wav file to be analyzed
``` 

# Why?
Why did I do this? As you may have noticed from the references below, there are already a couple of implementations, even one in Python! While I found these resources helpful in understanding the original [paper](http://soundlab.cs.princeton.edu/publications/2001_amta_aadwt.pdf), I still struggled to understand exactly what parts of the code correllated to what aspects of the paper. What better way to learn than by doing! So here I've re-implemented the algorithm detailed in the paper mostly from scratch, and have annotated as much as I could such that the implementation is as clear as possbile. Hopefully you can learn as much as I did, without having to dig through code and text to understand what's going on!

# References:
[Audio Analysis using the Discrete Wavelet Transform](http://soundlab.cs.princeton.edu/publications/2001_a
mta_aadwt.pdf)<br>
[Discrete Wavelet Transform Info](https://en.wikipedia.org/wiki/Discrete_wavelet_transform)<br>
[Marco Ziccardi's Blog & Scala Implementation](http://mziccard.me/2015/06/12/beats-detection-algorithms-2/)<br>
[Scaprot's own Python Implementation](https://github.com/scaperot/the-BPM-detector-python/blob/master/bpm_detection/bpm_detection.py)<br>
