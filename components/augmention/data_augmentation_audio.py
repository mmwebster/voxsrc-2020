# ### Data Augmentation for WAV Files - VoxSRC 2020

# #### To install all the required dependencies

get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install librosa')

import numpy as np # To use numpy data manipulation arrays
import librosa # To do audio manipulation
from matplotlib import pyplot as plt # To plot data
import librosa.display # To display MFCC Spectograms
import IPython.display as ipd # To listen to audio clips in notebook
# import pyrubberband # To change speed and pitch of audio clips
import soundfile as sf # To write audio files
get_ipython().magic('matplotlib inline')

# Reading the WAV file and loading it in as a floating point time series
def readWAVFile(filename):
    signal, sample_rate = librosa.load(filename)
    return signal, sample_rate

def addWhiteNoise(signal, sample_rate, req_snr):
    # https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
    RMS_signal = np.sqrt(np.mean(signal**2))
    RMS_noisy_sig = np.sqrt(RMS_signal**2/(10 ** (req_snr/10)))
    noise = np.random.normal(0, RMS_noisy_sig, signal.shape[0])
    whiteNoisySignal = signal + noise
    return whiteNoisySignal

def addBabble(signal, sample_rate):
    return babbleWavFile

def addTimeDelay(signal, sample_rate):
    return timeDelayFileArray

def increasePitch(signal, sample_rate):
    return increasePitchWavFile

def decreasePitch(signal, sample_rate):
    return decreasePitchWavFile

def changeSpeed(signal, sample_rate, speed_factor):
    changeSpeedSignal = librosa.core.resample(signal, sample_rate, 1/speed_factor * sample_rate)
    return changeSpeedSignal

def convertToMFCC(signal, sample_rate):
    
    # Converting signal to MFCC features
    mfcc_features = librosa.feature.mfcc(y=signal, sr=sample_rate)
    
    # Displaying the MFCC - sanity check
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_features, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    
    return mfcc_features

def saveMFCC(MFCC):
    return True

def saveWAV(signal, sample_rate, filepath):
    sf.write(filepath, signal, sample_rate, format='wav')
