import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, windows
from scipy.fft import fft

class OCTActuator:
    def __init__(self):
        self.waveform_command = np.array([])
        self.waveform_NICardOutput = np.array([])
        self.waveform_coil = np.array([])
        self.waveform_coil_valid = False
        self.amplifier_gain_dB = 0
        self.frequency_expected = 0
        self.Fs = 0
        self.session_name = ""
        self.frequency = 0
        self.amplitude_command_v = 0
        self.amplitude_v = 0
        self.amplitude_um = 0
        self.freq_content = np.array([])
        self.freq_range = np.array([])

    def compute_waveforms(self, folder=".", Fs=0, session_name="", manual_check=False, amplifier_gain_dB=None, SNR_check=True):
        self.Fs = Fs
        self.session_name = session_name
        if amplifier_gain_dB is not None:
            self.amplifier_gain_dB = amplifier_gain_dB 
        self.load_waveforms(folder)
        self.compute_amplitude(data_select="command", SNR_check=SNR_check, manual_check=manual_check)
        self.compute_amplitude(data_select="coil", SNR_check=SNR_check, manual_check=manual_check)

    def apply_downsample(self, nsample_target):
        nsample_origin = len(self.waveform_command)
        interval = np.round(np.linspace(0, nsample_origin, nsample_target + 1)).astype(int)
        self.waveform_command = np.array([np.mean(self.waveform_command[start:stop]) for start, stop in zip(interval[:-1], interval[1:])])
        self.waveform_NICardOutput = np.array([np.mean(self.waveform_NICardOutput[start:stop]) for start, stop in zip(interval[:-1], interval[1:])])
        self.waveform_coil = np.array([np.mean(self.waveform_coil[start:stop]) for start, stop in zip(interval[:-1], interval[1:])])
        self.Fs = self.Fs * (len(self.waveform_command) / nsample_origin)

    def load_waveforms(self, folder="."):
        folder = str(folder)
        header_offset = 3
        sampled_data = pd.read_csv(f"{folder}/SampledData", sep='\t')
        NICardOutput = sampled_data["StimulusGenBoard/ai0"][header_offset:].values
        command = np.loadtxt(f"{folder}/waveform.txt")
        amplification_ratio = 10 ** (self.amplifier_gain_dB / 20)
        self.waveform_command = command * amplification_ratio
        self.waveform_NICardOutput = np.array(NICardOutput, dtype=float) * amplification_ratio
        self.waveform_coil = sampled_data["StimulusGenBoard/ai1"][header_offset:].values

        coil = self.waveform_coil
        coil = np.array(coil, dtype=float)
        if np.abs(np.mean(coil)) > 1e-1:
            # it should be roughly centered toward 0V, if not called it invalid
            self.waveform_coil_valid = False
        else:
            self.waveform_coil_valid = True

    def show(self):
        plt.plot(self.waveform_command, label='Initial command')
        plt.plot(self.waveform_NICardOutput, label='NI output')
        plt.plot(self.waveform_coil, label='amplified')
        plt.title(self.session_name)
        plt.legend()
        plt.show(block=True)

    def get_period_idx(self, data_select="coil", smoothing=True, show=False, title=None, save=False, save_path=""):
        trajectory = getattr(self, f"waveform_{data_select}")
        nsample_per_period_expected = self.Fs / self.frequency
        nsample = len(trajectory)
        if nsample / nsample_per_period_expected < 3:
            return np.array([[0, nsample]])
        if smoothing:
            window = 1e-2 * (self.Fs / self.frequency)
            if window < 1:
                window = 1
            else:
                window = int(window)
            trajectory = pd.Series(trajectory).rolling(window=window).mean().values
        pos_to_neg_idx = np.where((trajectory[:-1] > 0) & (trajectory[1:] < 0))[0]
        pos_to_neg_idx = np.concatenate(([0], pos_to_neg_idx, [len(trajectory) - 1]))
        nsample_per_period = np.diff(pos_to_neg_idx)
        nsample_ratio = np.abs((nsample_per_period - nsample_per_period_expected) / nsample_per_period_expected)
        nsample_valid_idx = nsample_ratio < 1e-2
        # if no valid period index (close to expected number of sample), return the whole signal
        if sum(nsample_valid_idx==True) < 2:
            return np.array([[0, nsample]])

        start_period_idx = pos_to_neg_idx[np.append(nsample_valid_idx, False)]
        nsample_detected = int(np.round(np.mean(np.diff(start_period_idx))-1))
        stop_period_idx = start_period_idx + nsample_detected
        tuples = np.column_stack((start_period_idx, stop_period_idx))
        amplitudes = np.array([max(trajectory[start:stop])-min(trajectory[start:stop]) for start, stop in tuples])
        upper_limit = np.mean(amplitudes) + np.std(amplitudes)
        lower_limit = np.mean(amplitudes) - np.std(amplitudes)
        tuples = tuples[(amplitudes > lower_limit) & (amplitudes < upper_limit)]
        if show:
            cmap = plt.get_cmap("jet", len(tuples))
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(trajectory, 'k')
            for t, (start, stop) in enumerate(tuples):
                plt.plot(range(start, stop), trajectory[start:stop], color=cmap(t), linewidth=2)
            plt.title('Signal with Highlighted Segments')
            plt.subplot(2, 1, 2)
            for start, stop in tuples:
                plt.plot(trajectory[start:stop], 'k')
            plt.title('Segments Overlapped')
            plt.suptitle(title if title else self.session_name)
            if save:
                plt.savefig(save_path, dpi=500)
            plt.show()
        return tuples

    def compute_amplitude(self, frequency_expected=-1, data_select="coil", data_custom=None, fullperiod=True, usedetrend=True, windowing=True, zeropadding=True, SNR_check=True, SNR=2, verbose=True, manual_check=False):
        trajectory = getattr(self, f"waveform_{data_select}") if data_select != "custom" else data_custom
        if fullperiod:
            nsamples = len(trajectory)
            if frequency_expected == -1:
                self.compute_amplitude(data_select="command", fullperiod=False, verbose=False)
                frequency_expected = self.frequency
                self.frequency = 0
                self.freq_range = np.array([])
                self.amplitude_v = np.nan
                self.amplitude_um = np.nan
            if frequency_expected > 15:
                print("breakpoint location for debugging...")
            nsamples_per_period = self.Fs / frequency_expected
            pids = np.floor(np.arange(nsamples_per_period, nsamples, nsamples_per_period)).astype(int)
            if len(pids) > 1:
                trajectory = trajectory[pids[0]:pids[-1]]
        nsamples = len(trajectory)
        if usedetrend:
            trajectory = detrend(trajectory)
        if windowing:
            trajectory *= windows.hann(nsamples)
        nFFT = int(self.Fs) if zeropadding else nsamples
        P2 = fft(trajectory, nFFT) / nsamples
        if windowing:
            P2 *= 2
        P2 = np.abs(P2[:nFFT // 2 + 1])
        P2[1:] *= 2
        f_resolution = self.Fs / nFFT
        self.freq_range = np.linspace(0, self.Fs / 2 - f_resolution, nFFT // 2 + 1)
        fstim_id = np.argmax(P2) if frequency_expected == -1 else np.argmin(np.abs(self.freq_range - frequency_expected))
        self.frequency = self.freq_range[fstim_id]
        if SNR_check:
            idx_lowpass = np.argmin(np.abs(self.freq_range - self.frequency * 0.9))
            idx_highpass = np.argmin(np.abs(self.freq_range - self.frequency * 1.1))
            safezone = np.arange(max(0, idx_lowpass), min(idx_highpass, len(self.freq_range)))
            not_fstim_freq = np.setdiff1d(np.arange(len(self.freq_range)), safezone)
            if P2[fstim_id] < max(P2[not_fstim_freq]) * SNR:
                return np.nan
        amp = P2[fstim_id]
        if verbose:
            print(f"Type({data_select}), Frequency = {self.frequency} Hz, Amplitude peak = {amp} V.")
        if data_select == "coil":
            self.amplitude_v = amp
        elif data_select == "command":
            self.amplitude_command_v = amp
        self.amplitude_um = np.nan
        if manual_check:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            ax.plot(self.waveform_command, label='Command signal (V)')
            if len(self.waveform_coil) > 0:
                ax.plot(self.waveform_coil, label='Coil (V)')
            ax.plot(trajectory, label='FFT input signal')
            ax.legend()
            ax.set_title("time domain")
            fmax = min(1000, self.frequency * 5)
            fmax_id = np.argmin(np.abs(self.freq_range - fmax))
            ax.plot(self.freq_range[:fmax_id], P2[:fmax_id])
            ax.text(0.8 * self.frequency, amp / 2, f'Peak val: {amp:.4f}', fontsize=12)
            ax.set_title("frequency domain")
            plt.suptitle('OCT STIMULI: compute_amplitude // Manual control')
            plt.show()
        return amp