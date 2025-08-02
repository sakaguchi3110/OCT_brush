import os
import numpy as np
import h5py
from datetime import datetime
from PIL import Image

import matplotlib
# Check if running on SSH server
if 'SSH_CONNECTION' in os.environ:
    matplotlib.use('TkAgg')  # Use TkAgg or any other X11 back-end
else:
    pass
    #matplotlib.use('Agg')  # Use a non-interactive back-end for local use or environments without X11
import matplotlib.pyplot as plt


class OCTMetadata:
    IDLETYPE = 'IDLE'
    STIMTYPE = 'H5T_STD_U16LE'
    OCTCCD_NPIXELS = 2048
    VIDEOFNAME = 'Video.jpg'
    MEASFILE_INFO = 'Measurement.srm'
    MEASFILE_DATA = 'MeasurementShort'
    MEASFILE_STRUCT2D = 'Measurement.raw'
    MEASFILE_STRUCT3D = 'Measurement'

    def __init__(self):
        self.folder = ''
        self.session_name = ''
        self.downsample = False
        self.downsample_method = float('nan')
        self.nsample = float('nan')
        self.nsample_original = float('nan')
        self.Fs_OCT = float('nan')
        self.Fs_data = float('nan')
        self.nFFTFlag = 'all'
        self.Apo = float('nan')
        self.Waveform = float('nan')
        self.K = float('nan')
        self.KES = float('nan')
        self.dataType = ''
        self.isVibration = False
        self.isStructural = False
        self.whichStructural = None
        self.Video = np.array([])
        self.dasp = float('nan')
        self.Av = float('nan')
        self.n_alines = float('nan')
        self.Volt = float('nan')
        self.StimFreq = float('nan')
        self.Length = float('nan')
        self.Xstart = float('nan')
        self.Ystart = float('nan')
        self.Xstop = float('nan')
        self.Ystop = float('nan')
        self.timestamp = ''

    def load(self, folder=os.getcwd(), nFFTFlag='all'):
        self.folder = folder
        self.nFFTFlag = nFFTFlag
        self.session_name = os.path.basename(self.folder)
        self.extract_measInfo()
        self.extract_stimulusCmd()
        self.define_dataType()
        self.extract_apodization()
        self.coeff_linearity_waves()

    def extract_measInfo(self):
        # parts = self.folder.split('\\')
        # parts[-2] = parts[-2].replace('_block', '_OVP_block')
        # new_folder = '\\'.join(parts) # with open(os.path.join(self.folder, self.MEASFILE_INFO), 'r') as fid:
        # with open("Z:/OCT_VIB_PSYCHOMETRIC/1_primary/Ptest/data_oct/2025-05-05_11-50-17_P001_OVP_block-01_trial-01/interval2/Measurement.srm") as fid:
        with open(os.path.join(self.folder, self.MEASFILE_INFO).replace(os.sep,'/'), 'r') as fid:
            meas_content = fid.read()

        self.Video = np.array(Image.open(os.path.join(self.folder, self.VIDEOFNAME)))
        self.dasp = []
        self.nsample_original = 0
        self.timestamp = 'timestamp unknown'

        lines = meas_content.splitlines()
        for line in lines:
            r = line.split(':')
            if len(r) < 2:
                continue
            id = r[0]
            val = r[1].replace(',', '.')

            if id.startswith("Averages"):
                self.Av = float(val)
            elif id.startswith("Samples"):
                self.nsample_original = int(val)
            elif id.startswith("Width"):
                self.n_alines = int(val)
            elif id.startswith("Stim. Amp"):
                self.Volt = float(val)
            elif id.startswith("Stim. freq."):
                self.StimFreq = float(val)
            elif id.startswith("Length"):
                self.Length = float(val)
            elif id.startswith("Xstart"):
                self.Xstart = float(val)
            elif id.startswith("Ystart"):
                self.Ystart = float(val)
            elif id.startswith("Xstop"):
                self.Xstop = float(val)
            elif id.startswith("Ystop"):
                self.Ystop = float(val)
            elif id.startswith("Timestamp"):
                self.timestamp = ':'.join(r[1:]).strip()
            elif id.startswith("Sample Rate (Hz)"):
                self.Fs_OCT = float(val)

        try:
            Xpos = np.loadtxt(os.path.join(self.folder, 'Xpositions.txt'))
            self.Xstart = Xpos
            self.Xstop = Xpos[-1]
            Ypos = np.loadtxt(os.path.join(self.folder, 'Ypositions.txt'))
            self.Ystart = Ypos
            self.Ystop = Ypos[-1]
        except:
            correction = -240 / 56
            self.Xstart += correction
            self.Ystart += correction
            self.Xstop += correction
            self.Ystop += correction

        if not self.Fs_OCT or np.isnan(self.Fs_OCT):
            if self.timestamp != 'timestamp unknown':
                # Correct format to match the full timestamp
                recording_datetime = datetime.strptime(self.timestamp, '%Y-%m-%d %H:%M:%S')
                if recording_datetime < datetime(2024, 8, 1):
                    self.Fs_OCT = 9454.65
                else:
                    self.Fs_OCT = 147e3
            else:
                self.Fs_OCT = 147e3

        if not self.downsample:
            self.Fs_data = self.Fs_OCT
            self.nsample = self.nsample_original

    def extract_stimulusCmd(self):
        txtFiles = [f for f in os.listdir(self.folder) if f.endswith('.txt') and 'positions.txt' not in f]
        self.Waveform = []
        for txtFile in txtFiles:
            waveform = np.loadtxt(os.path.join(self.folder, txtFile))
            if len(waveform) >= 500:
                self.Waveform = waveform
                break

    def define_dataType(self):
        if os.path.isfile(os.path.join(self.folder, self.MEASFILE_DATA)):
            with h5py.File(os.path.join(self.folder, self.MEASFILE_DATA), 'r') as f:
                #self.dataType = f['RawSpectra'].dtype
                self.dataType = 'H5T_STD_U16LE'
            self.isStructural = False
            self.whichStructural = None
        elif os.path.isfile(os.path.join(self.folder, self.MEASFILE_STRUCT2D)):
            self.dataType = self.IDLETYPE
            self.isStructural = True
            self.whichStructural = "2D"
        elif os.path.isfile(os.path.join(self.folder, self.MEASFILE_STRUCT3D)):
            self.dataType = self.IDLETYPE
            self.isStructural = True
            self.whichStructural = "3D"
        self.isVibration = not self.isStructural

    def extract_apodization(self):
        try:
            with h5py.File(os.path.join(self.folder, 'ApodizationData'), 'r') as f:
                self.Apo = f['/Apodization'][:]
        except:
            with open(os.path.join(self.folder, 'ApodizationData'), 'rb') as fid:
                self.Apo = np.fromfile(fid, dtype='float32')

    def coeff_linearity_waves(self):
        KES_tmp = np.zeros(self.OCTCCD_NPIXELS)
        coeff = [-7.80145203693039e-10, 
                 3.14375722610804e-06, 
                 0.0510905679348588, 
                 -554.441673073287, 
                 5293102.26158046]
        K_tmp = np.polyval(coeff, np.arange(1, self.OCTCCD_NPIXELS + 1))
        Kstep = (K_tmp[-1] - K_tmp[0]) / (self.OCTCCD_NPIXELS - 1)

        for i in range(self.OCTCCD_NPIXELS):
            KES_tmp[i] = K_tmp[0] + Kstep * i

        self.K = K_tmp
        self.KES = KES_tmp
        if 0:
            plt.plot(self.K)
            plt.plot(self.KES)
            plt.show()
