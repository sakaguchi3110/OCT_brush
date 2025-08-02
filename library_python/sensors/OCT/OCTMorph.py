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
# Check if running on SSH server
if 'SSH_CONNECTION' in os.environ:
    matplotlib.use('TkAgg')  # Use TkAgg or any other X11 back-end
else:
    pass
    #matplotlib.use('Agg')  # Use a non-interactive back-end for local use or environments without X11
import matplotlib.pyplot as plt


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
                    # Trim the array if it's too large
                    morph_ = morph_[:np.prod(expected_shape)]
                elif morph_.size < np.prod(expected_shape):
                    # Pad the array with zeros if it's too small
                    morph_ = np.pad(morph_, (0, np.prod(expected_shape) - morph_.size))
                self.morph = morph_.reshape(expected_shape)
            
            self.morph_ampl = np.abs(self.morph)
            # decibel adjustment has been already made on LabView:
            self.morph_dB_img = self.morph_ampl
            # self.save_morph_dB_img()
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

    # if downsample is applied on already saved data, there is a chance that only .morph exists.
    # create verifications to match existing object characteristics
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
        if adjust_inf == True:
            # Modify values below 1 to 1
            morph_ampl[morph_ampl<1] = 1

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

def hardware_correction(dataSlice, metadata):
    # Ensure dataSlice is a 2D array (squeeze if necessary)
    dataSlice = np.squeeze(dataSlice)
    nsample = dataSlice.shape[-1]
    # If the data type is STIMTYPE, apply the scaling factor
    if metadata.dataType == metadata.STIMTYPE:
        dataSlice *= 540
    
    # Sort indices based on the metadata.K values
    sorted_indices = np.argsort(metadata.K)
    # Initialize the corrected array with the same shape as dataSlice
    corrected = np.zeros_like(dataSlice)
    # Perform the correction for each sample
    for i in range(nsample):
        # Subtract the Apo value from each column (sample) of the data
        dataSlice[:, i] -= metadata.Apo
        # Create the cubic spline interpolation using sorted K values
        cs = CubicSpline(metadata.K[sorted_indices], dataSlice[sorted_indices, i], axis=0)
        # Interpolate at the desired points (KES) and store in the corrected array
        corrected[[sorted_indices], i] = cs(metadata.KES[sorted_indices])
    
    # sum(sum(abs(corrected-dataSlice))) / sum(sum(abs(corrected))) can shows 40% difference
    # similar results on Matlab side
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