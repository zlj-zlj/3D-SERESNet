import numpy as np
from pyedflib import highlevel
from scipy import signal
def apply_filters(signals, sampling_rate):
    num_channels = signals.shape[0]
    filtered_signals = np.zeros_like(signals)

    for channel in range(num_channels):
        f0 = 60.0
        Q = 30.0
        w0 = f0 / (sampling_rate / 2)
        b, a = signal.iirnotch(w0, Q)
        signals_notch = signal.filtfilt(b, a, signals[channel, :])

        lowcut = 0.5
        highcut = 100.0
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signals[channel, :] = signal.filtfilt(b, a, signals_notch)

    return filtered_signals


def process_edf_fixed_params(edf_path, output_path=None):


    window_size = 10
    sampling_rate = 256
    step_size = 10
    window_samples = window_size * sampling_rate
    step_samples = step_size * sampling_rate


    signals, signal_headers, header = highlevel.read_edf(edf_path)

    signals = np.array(signals)


    num_channels, num_samples = signals.shape
    sliced_data = []

    for start in range(0, num_samples - window_samples + 1, step_samples):
        end = start + window_samples
        sliced_data.append(signals[:, start:end])

    sliced_data = np.array(sliced_data)


    if output_path:

        new_signal_headers = signal_headers[:num_channels]
        new_header = header.copy()
        new_signals = sliced_data.reshape(num_channels, -1)
        highlevel.write_edf(output_path, new_signals, new_signal_headers, new_header)

    return sliced_data


def normalize_within_sample(samples):
    mean = np.mean(samples, axis=-1, keepdims=True)
    std = np.std(samples, axis=-1, keepdims=True)
    normalized_samples = (samples - mean) / std
    return normalized_samples
