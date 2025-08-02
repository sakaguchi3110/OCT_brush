import numpy as np
import matplotlib.pyplot as plt

class OCTTrialIdentifier:
    width_pulse = 5  # ms
    inter_pulse = 5  # ms
    between_freq_part = 30  # ms

    def __init__(self, data=None, Fs=0, session_name=""):
        self.data = data if data is not None else np.array([])
        self.Fs = Fs
        self.session_name = session_name
        self.frequency = 0
        self.trial_number = 0

    def compute_waveform(self, folder='.', Fs=0, session_name="", info_extract_method="fromName"):
        self.Fs = Fs
        self.session_name = session_name
        self.load_waveform(folder)

        if info_extract_method == "fromName":
            self.get_trial_identifier_info_from_name()
        elif info_extract_method == "fromSignal":
            self.get_trial_identifier_info_from_signal()

    def get_trial_identifier_info_from_name(self):
        parts = self.session_name.split('_')
        self.trial_number = float(parts)
        parts = parts.split('Hz')
        self.frequency = float(parts)

    def get_trial_identifier_info_from_signal(self):
        nsample_min_between_part = int(self.Fs * self.between_freq_part / 1e3 * 0.95)
        idx_p2 = self.find_long_zero_chain(self.data, nsample_min_between_part)
        idx_p3_offset = self.find_long_zero_chain(self.data[idx_p2 + nsample_min_between_part:], nsample_min_between_part)
        idx_p3 = idx_p2 + nsample_min_between_part + idx_p3_offset
        idx_p3_end_offset = self.find_long_zero_chain(self.data[idx_p3 + nsample_min_between_part:], nsample_min_between_part)
        idx_p3_end = idx_p3 + nsample_min_between_part + idx_p3_end_offset

        digit = np.sum(self.data[:idx_p2])
        self.frequency = 10 ** (digit - 1) * np.sum(self.data[idx_p2:idx_p3])
        self.trial_number = np.sum(self.data[idx_p3:idx_p3_end])

    def load_waveform(self, folder='.'):
        self.data = np.loadtxt(f"{folder}/identifier.txt")

    def show_waveforms(self):
        plt.plot(self.data, label='identifier')
        plt.legend()
        plt.show()

    def fprintflocal(self, s):
        print(f"OCTTrialIdentifier :: {s}")

    @staticmethod
    def find_long_zero_chain(data, min_length):
        zero_chains = np.where(data == 0)
        diffs = np.diff(zero_chains)
        long_zero_chains = np.where(diffs > min_length)
        if len(long_zero_chains) > 0:
            return zero_chains[long_zero_chains]
        return -1