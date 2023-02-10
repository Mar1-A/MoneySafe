import numpy as np
import scipy.fftpack
import scipy.signal


def mfcc(
    signal,
    sample_rate=16000,
    num_cepstra=13,
    nfilt=26,
    nfft=512,
    low_freq=0,
    high_freq=None,
):
    """
    Calculates the Mel-Frequency Cepstral Coefficients (MFCCs) of a given signal.
    :param signal: The audio signal (1-dimensional numpy array)
    :param sample_rate: The sample rate of the signal (in Hz)
    :param num_cepstra: The number of cepstra to return
    :param nfilt: The number of filters in the filterbank
    :param nfft: The FFT size
    :param low_freq: Lowest frequency in filterbank (in Hz)
    :param high_freq: Highest frequency in filterbank (in Hz), or None for Nyquist
    :return: A 2-dimensional numpy array, where each row is the MFCC of the corresponding frame.
    """
    if high_freq is None:
        high_freq = sample_rate // 2

    # Pre-emphasis filtering
    pre_emphasis = 0.97
    signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = int(0.025 * sample_rate)
    frame_stride = int(0.01 * sample_rate)
    frames = np.array(
        [
            signal[i : i + frame_size]
            for i in range(0, len(signal) - frame_size + 1, frame_stride)
        ]
    )

    # Window function
    frames *= np.hamming(frame_size)

    # FFT
    complex_spectrum = np.array([scipy.fftpack.fft(frame, nfft) for frame in frames])
    magnitude_spectrum = np.abs(complex_spectrum)

    # Filterbank
    high_mel = hz2mel(high_freq)
    low_mel = hz2mel(low_freq)
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel2hz(mel_points)
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            filter_banks = np.dot(magnitude_spectrum, fbank.T)
            filter_banks = np.where(
                filter_banks == 0, np.finfo(float).eps, filter_banks
            )  # Numerical Stability
            filter_banks = 20 * np.log10(filter_banks)  # dB

        # DCT
        mfcc = scipy.fftpack.dct(filter_banks, axis=1, type=2, norm="ortho")[
            :, :num_cepstra
        ]

        # Lifter
        cep_lifter = 22
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift

    return mfcc


def hz2mel(hz):
    """
    Converts a frequency in Hz to its equivalent in Mels
    :param hz: The frequency in Hz
    :return: The equivalent frequency in Mels
    """
    return 2595 * np.log10(1 + hz / 700)


def mel2hz(mel):
    """
    Converts a frequency in Mels to its equivalent in Hz
    :param mel: The frequency in Mels
    :return: The equivalent frequency in Hz
    """
    return 700 * (10 ** (mel / 2595) - 1)
