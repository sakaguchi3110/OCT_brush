import numpy as np
import h5py
import os
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from scipy.signal import detrend, butter, filtfilt
from scipy.signal.windows import hann
from scipy.interpolate import CubicSpline
from PIL import Image
from pathlib import Path

import matplotlib
if 'SSH_CONNECTION' in os.environ:
    matplotlib.use('TkAgg')
else:
    pass
import matplotlib.pyplot as plt

# === Temporal averaging helpers (minimal) ===
def _as_linear_from_db_amp(I_dB: np.ndarray) -> np.ndarray:
    """20*log10 振幅dB → 線形振幅"""
    # ✅ dtype引数を使わずastypeでキャスト（np.powerにdtypeは無い）
    return np.power(10.0, I_dB / 20.0).astype(np.float64)

def _bulk_phase_align(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """参照 ref と x のバルク位相差を推定して補正（安全最小限）"""
    dphi = np.angle(np.vdot(ref.ravel(), x.ravel()))
    return x * np.exp(-1j * dphi)



class OCTMorph:
    def __init__(self, metadata=None, downsample_method=1):
        self.md = metadata
        self.downsample_method = downsample_method
        self.raw = []
        self.corrected = []
        self.morph = []
        self.morph_ampl = []
        self.morph_dB_img = []
        self.morph_dB_video = []

    def compute_morph(self):
        if self.md.isStructural:
            ndepths = int(self.md.OCTCCD_NPIXELS/2)
            f = ""
            if self.md.whichStructural == "2D":
                f = open(f"{self.md.folder}/{self.md.MEASFILE_STRUCT2D}", 'rb')
                morph_ = np.fromfile(f, dtype=np.float32)
                n_alines = morph_.size // ndepths
                self.morph = morph_.reshape((ndepths, n_alines))
            elif self.md.whichStructural == "3D":
                f = open(f"{self.md.folder}/{self.md.MEASFILE_STRUCT3D}", 'rb')
                ncols = morph_.size // ndepths
                n_blines = int(ncols / self.md.n_alines)
                expected_shape = (ndepths, self.md.n_alines, n_blines)
                
                if morph_.size == np.prod(expected_shape):
                    pass
                elif morph_.size > np.prod(expected_shape):
                    morph_ = morph_[:np.prod(expected_shape)]
                elif morph_.size < np.prod(expected_shape):
                    morph_ = np.pad(morph_, (0, np.prod(expected_shape) - morph_.size))
                self.morph = morph_.reshape(expected_shape)
            
            self.morph_ampl = np.abs(self.morph)
            self.morph_dB_img = self.morph_ampl
        else:
            self.load_raw_data()
            self.apply_hardware_correction()
            self.create_morph()
            self.get_morph_img()
            self.get_morph_video()
        if self.md.isVibration and self.md.downsample and self.downsample_method == 3:
            self.apply_downsample(self.md.nsample)
        print("Morph is done.")

    def get_nsample(self):
        nsample = None
        if len(self.raw):
            nsample = self.raw.shape[-1]
        elif len(self.morph):
            nsample = self.morph.shape[-1]
        elif len(self.morph_ampl):
            nsample = self.morph_ampl.shape[-1]
        elif len(self.morph_dB_video):
            nsample = self.morph_dB_video.shape[-1]
        return nsample

    def apply_downsample(self, nsample_target):
        if len(self.morph) == 0:
            self.create_morph()
        if len(self.morph_ampl) == 0:
            self.morph_ampl = np.abs(self.morph)
        if len(self.morph_dB_video) == 0:
            self.get_morph_video()

        nLines, nDepths = self.morph.shape[:2]
        nCCDPixel = 2 * nDepths
        if len(self.raw):
            nsample_origin = self.raw.shape[-1]
        else:
            nsample_origin = self.morph.shape[-1]
        interval = np.round(np.linspace(1, nsample_origin, nsample_target + 1)).astype(int)

        raw_ = np.zeros((nLines, nCCDPixel, nsample_target), dtype=np.complex64)
        corrected_ = np.zeros((nLines, nCCDPixel, nsample_target), dtype=np.complex64)
        morph_ = np.zeros((nLines, nDepths, nsample_target), dtype=np.complex64)
        morph_ampl_ = np.zeros((nLines, nDepths, nsample_target), dtype=np.float32)
        morph_dB_video_ = np.zeros((nLines, nDepths, nsample_target))

        for t in range(len(interval) - 1):
            R = slice(interval[t], interval[t + 1])
            if len(self.raw) > 0:
                raw_[:, :, t] = np.mean(self.raw[:, :, R], axis=2)
            if len(self.corrected) > 0:
                corrected_[:, :, t] = np.mean(self.corrected[:, :, R], axis=2)
            morph_[:, :, t] = np.mean(self.morph[:, :, R], axis=2)
            morph_ampl_[:, :, t] = np.mean(np.abs(self.morph_ampl[:, :, R]), axis=2)
            morph_dB_video_[:, :, t] = np.mean(np.abs(self.morph_dB_video[:, :, R]), axis=2)

        self.raw = raw_
        self.corrected = corrected_
        self.morph = morph_
        self.morph_ampl = morph_ampl_
        self.morph_dB_video = morph_dB_video_

    def load_raw_data(self):
        nLines = self.md.n_alines
        nCCDPixel = self.md.OCTCCD_NPIXELS
        nSamples = self.md.nsample_original
        raw_ = np.zeros((nLines, nCCDPixel, nSamples), dtype=np.float32)
        fname = f"{self.md.folder}/{self.md.MEASFILE_DATA}"
        with h5py.File(fname, 'r') as f:
            for aline_id in range(nLines):
                print(f"Morph extracting ({aline_id + 1}/{nLines})")
                if aline_id < f['RawSpectra'].shape[0]:
                    data = np.transpose(f['RawSpectra'][aline_id, :, :])
                else:
                    data = np.zeros((nCCDPixel, nSamples))
                raw_[aline_id, :, :] = data

        if self.md.downsample and self.downsample_method == 1:
            self.raw = np.zeros((nLines, nCCDPixel, self.md.nsample), dtype=np.float32)
            interval = np.round(np.linspace(0, self.md.nsample_original-1, self.md.nsample)).astype(int)
            for t in range(len(interval)):
                self.raw[:, :, t] = raw_[:, :, interval[t]]
        elif self.md.downsample and self.downsample_method == 2:
            self.raw = np.zeros((nLines, nCCDPixel, self.md.nsample), dtype=np.float32)
            interval = np.round(np.linspace(0, self.md.nsample_original-1, self.md.nsample)).astype(int)
            for t in range(len(interval) - 1):
                start, stop = interval[t], interval[t + 1]
                self.raw[:, :, t] = np.mean(raw_[:, :, start:stop], axis=2)
        else:
            self.raw = raw_

    def apply_hardware_correction(self):
        nLines = self.md.n_alines
        nCCDPixel = self.md.OCTCCD_NPIXELS
        nSamples = self.raw.shape[-1]
        corrected_ = np.zeros((nLines, nCCDPixel, nSamples), dtype=np.float32)
        for aline_id in range(nLines):
            print(f"Morph hardware_correction ({aline_id + 1}/{nLines})")
            corrected_[aline_id, :, :] = hardware_correction(self.raw[aline_id, :, :], self.md)
        self.corrected = corrected_

    def create_morph(self):
        nLines = self.md.n_alines
        nDepths = self.md.OCTCCD_NPIXELS // 2
        nSamples = self.corrected.shape[-1]
        morph_ = np.zeros((nLines, nDepths, nSamples), dtype='complex128')
        for aline_id in range(nLines):
            print(f"Morph fft_slice ({aline_id + 1}/{nLines})")
            morph_[aline_id, :, :] = fft_slice(self.corrected[aline_id, :, :])
        self.morph = morph_
        self.morph_ampl = np.abs(self.morph)

    def get_morph_video(self, adjust_inf=True, verbose=False):
        self.morph_dB_video = []
        if len(self.morph_ampl) == 0:
            self.morph_ampl = np.abs(self.morph)
        morph_ampl = self.morph_ampl
        if adjust_inf:
            morph_ampl[morph_ampl < 1] = 1
        nLines = morph_ampl.shape[0]
        for aline_id in range(nLines):
            if verbose:
                print(f"Morph video processing ({aline_id + 1}/{nLines})")
            self.morph_dB_video.append(get_morph_video(morph_ampl[aline_id, :, :]))
        self.morph_dB_video = np.array(self.morph_dB_video)

    def get_morph_img(self, verbose=False):
        if len(self.morph_ampl) == 0:
            if len(self.morph) == 0:
                self.create_morph()
            self.morph_ampl = np.abs(self.morph)
        nLines = self.morph_ampl.shape[0]
        for aline_id in range(nLines):
            if verbose:
                print(f"Morph structural image processing ({aline_id + 1}/{nLines})")
            if self.md.isStructural:
                self.morph_dB_img.append(get_morph_img(self.morph_ampl[aline_id, :]))
            else:
                self.morph_dB_img.append(get_morph_img(self.morph_ampl[aline_id, :, :]))
        self.morph_dB_img = np.array(self.morph_dB_img)
        
    # --- 追加メソッド ---
    def average_over_time(self, average_domain: str, time_mode: str, block_size: int = 10):
        """
        average_domain: "complex" or "amplitude"
        time_mode:      "full" or "reduced"
        戻り値: dB スタック（selfの配列は上書きしない）
        """
        assert average_domain in ("complex", "amplitude")
        assert time_mode in ("full", "reduced")

        if self.morph is None or len(self.morph) == 0:
            self.create_morph()
        if self.morph_dB_video is None or len(self.morph_dB_video) == 0:
            self.get_morph_video()

        nLines, nDepths, nSamples = self.morph.shape
        has_complex = np.iscomplexobj(self.morph)

        if average_domain == "complex" and has_complex:
            base_cplx = self.morph.astype(np.complex128)
        else:
            if self.morph_dB_video is not None and len(self.morph_dB_video) != 0:
                base_lin = _as_linear_from_db_amp(self.morph_dB_video)
            else:
                base_lin = np.abs(self.morph).astype(np.float64)
            average_domain = "amplitude"

        radius = 2  # スライディング窓（full）

        if time_mode == "full":
            out_dB = np.empty((nLines, nDepths, nSamples), dtype=np.float32)
            for a in range(nLines):
                if average_domain == "amplitude":
                    for t in range(nSamples):
                        t0, t1 = max(0, t - radius), min(nSamples, t + radius + 1)
                        seg = base_lin[a, :, t0:t1]
                        ampl = np.mean(seg, axis=-1)
                        out_dB[a, :, t] = 20*np.log10(np.maximum(ampl, 1e-12))
                else:
                    for t in range(nSamples):
                        t0, t1 = max(0, t - radius), min(nSamples, t + radius + 1)
                        ref = base_cplx[a, :, t]
                        acc = np.zeros(nDepths, dtype=np.complex128); w = 0
                        for s in range(t0, t1):
                            x = _bulk_phase_align(ref, base_cplx[a, :, s])
                            acc += x; w += 1
                        ampl = np.abs(acc / max(w, 1))
                        out_dB[a, :, t] = 20*np.log10(np.maximum(ampl, 1e-12))
            return out_dB

        else:
            nB = nSamples // block_size
            out_dB = np.empty((nLines, nDepths, nB), dtype=np.float32)
            for a in range(nLines):
                if average_domain == "amplitude":
                    for b in range(nB):
                        t0, t1 = b*block_size, (b+1)*block_size
                        seg = base_lin[a, :, t0:t1]
                        ampl = np.mean(seg, axis=-1)
                        out_dB[a, :, b] = 20*np.log10(np.maximum(ampl, 1e-12))
                else:
                    for b in range(nB):
                        t0, t1 = b*block_size, (b+1)*block_size
                        ref = base_cplx[a, :, t0]
                        acc = np.zeros(nDepths, dtype=np.complex128); w = 0
                        for s in range(t0, t1):
                            x = _bulk_phase_align(ref, base_cplx[a, :, s])
                            acc += x; w += 1
                        ampl = np.abs(acc / max(w, 1))
                        out_dB[a, :, b] = 20*np.log10(np.maximum(ampl, 1e-12))
            return out_dB

# --- 既存の下位関数はそのまま ---
def hardware_correction(dataSlice, metadata):
    dataSlice = np.squeeze(dataSlice)
    nsample = dataSlice.shape[-1]
    if metadata.dataType == metadata.STIMTYPE:
        dataSlice *= 540
    sorted_indices = np.argsort(metadata.K)
    corrected = np.zeros_like(dataSlice)
    for i in range(nsample):
        dataSlice[:, i] -= metadata.Apo
        cs = CubicSpline(metadata.K[sorted_indices], dataSlice[sorted_indices, i], axis=0)
        corrected[[sorted_indices], i] = cs(metadata.KES[sorted_indices])
    return corrected

def fft_slice(correctedSlice, ignoreLowFreqs=False, highPassFreqID=10):
    correctedSlice = np.squeeze(correctedSlice)
    L, n_sample = correctedSlice.shape
    wind = np.tile(hann(L), (n_sample, 1)).T
    P2 = fft(correctedSlice * wind, axis=0) / L
    P2 *= 2
    if ignoreLowFreqs:
        P2[0:highPassFreqID, :] = 0
    P1 = 2 * P2[:L // 2, :]
    return P1

def get_morph_img(morph, depth_range=None):
    if len(morph.shape) == 2:
        morph = np.mean(np.abs(morph), axis=1)
    if depth_range:
        morph_dB_img = morph
        morph_dB_img[depth_range] = 20 * np.log10(morph[depth_range])
    else:
        morph_dB_img = 20 * np.log10(morph)
    return morph_dB_img

def get_morph_video(morph, depth_range=None):
    morph = np.abs(morph)
    if depth_range:
        morph_dB_video = morph
        if len(morph.shape) == 2:
            morph_dB_video[depth_range, :] = 20 * np.log10(morph[depth_range, :])
        else:
            morph_dB_video[depth_range] = 20 * np.log10(morph[depth_range])
    else:
        morph_dB_video = 20 * np.log10(morph)
    return morph_dB_video
