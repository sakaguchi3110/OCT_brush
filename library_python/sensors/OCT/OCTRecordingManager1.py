import os
import shutil
import numpy as np
import scipy.io as sio
import pickle

from ..OCT.OCTMetadata1 import OCTMetadata
from ..OCT.OCTMorph import OCTMorph
from ..OCT.OCTActuator import OCTActuator
from ..OCT.OCTTrialIdentifier import OCTTrialIdentifier
from ..OCT.postprocessing.OCT_phaseChange import OCTPhaseChange


class OCTRecordingManager:
    def __init__(self, scan_folder_raw=os.getcwd(), scan_folder_processed="", autosave=False):
        self.scan_folder_raw = scan_folder_raw
        self.scan_folder_processed = scan_folder_processed
        self.autosave = autosave

        self.winzoomTrial = None
        self.trial_identifier = None
        self.actuator = None
        self.accelerometer = None

        self.metadata = None
        self.morph = None

        self.PChange = None
        self.PResolved = None
        self.PWavelet = None
        self.PTrack = None
        self.velocity = None

        if self.scan_folder_processed and not os.path.exists(self.scan_folder_processed):
            os.makedirs(self.scan_folder_processed)

    def enable_downsample(self, nsample, downsample_method=1):
        self.metadata.downsample = True
        self.metadata.downsample_method = downsample_method
        self.metadata.nsample = nsample
        self.metadata.Fs_data = self.metadata.Fs_OCT * (nsample / self.metadata.nsample_original)

    def get_trial_identifier(self):
        return self.trial_identifier.frequency, self.trial_identifier.trial_number

    def set_neural_data(self, winzoomTrial, force_processing=True, save_hdd=None, destdir=None):
        if save_hdd is None:
            save_hdd = self.autosave
        if destdir is None:
            destdir = self.scan_folder_processed

        if not force_processing:
            imported, data_already_exist = self.loadifexist("winzoomTrial.pkl", destdir)
        else:
            data_already_exist = False

        if data_already_exist:
            self.winzoomTrial = imported
        else:
            self.winzoomTrial = winzoomTrial

        if not save_hdd or data_already_exist:
            return

        self.saveVariable("winzoomTrial.pkl", self.winzooymTrial, destdir)

    def load_metadata(self, force_processing=True, save_hdd=None, destdir=None):
        if save_hdd is None:
            save_hdd = self.autosave
        if destdir is None:
            destdir = self.scan_folder_processed

        if not force_processing:
            imported, data_already_exist = self.loadifexist("metadata.pkl", destdir)
        else:
            data_already_exist = False

        if data_already_exist:
            self.metadata = imported
        else:
            self.metadata = OCTMetadata()
            self.metadata.load(self.scan_folder_raw)

        if not save_hdd or data_already_exist:
            return

        self.saveVariable("metadata.pkl", self.metadata, destdir)

    def compute_trial_identifier(self, force_processing=True, save_hdd=None, destdir=None):
        if save_hdd is None:
            save_hdd = self.autosave
        if destdir is None:
            destdir = self.scan_folder_processed

        if not force_processing:
            imported, data_already_exist = self.loadifexist("trial_identifier.pkl", destdir)
        else:
            data_already_exist = False

        if data_already_exist:
            self.trial_identifier = imported
        else:
            self.trial_identifier = OCTTrialIdentifier()
            self.trial_identifier.compute_waveform(self.scan_folder_raw, self.metadata.Fs_OCT, self.metadata.session_name)

        if not save_hdd or data_already_exist:
            return

        self.saveVariable("trial_identifier.pkl", self.trial_identifier, destdir)


    def compute_actuator_waveform(self, force_processing=True, save_hdd=None, destdir=None, adjust_metadata=True, SNR_check=True, manual_check=False):
        if save_hdd is None:
            save_hdd = self.autosave
        if destdir is None:
            destdir = self.scan_folder_processed

        if not force_processing:
            imported, data_already_exist = self.loadifexist("actuator_waveform.pkl", destdir)
        else:
            data_already_exist = False

        if data_already_exist:
            self.actuator = imported
        else:
            amplification_dB = 0
            dB_filename = "stimuli_info_amplifier.txt"
            dB_filename_abs = self.find_fileabspath_from_node(dB_filename)
            if dB_filename_abs is not None and os.path.isfile(dB_filename_abs):
                with open(dB_filename_abs, 'r') as file:
                    for line in file:
                        if "amplification_dB" in line:
                            amplification_dB = float(line.strip().split(",")[1])
                            break
            else:
                amplification_dB = 0
            
            self.actuator = OCTActuator()
            self.actuator.compute_waveforms(self.scan_folder_raw, self.metadata.Fs_OCT, self.metadata.session_name, SNR_check=SNR_check, manual_check=manual_check, amplifier_gain_dB=amplification_dB)

            if self.metadata.downsample:
                self.actuator.apply_downsample(self.metadata.nsample)

            if adjust_metadata:
                self.metadata.StimFreq = round(self.actuator.frequency)
                self.metadata.Volt = self.actuator.amplitude_v
                if save_hdd:
                    self.saveVariable("metadata.pkl", self.metadata, destdir)

        if not save_hdd or data_already_exist:
            return

        self.saveVariable("actuator_waveform.pkl", self.actuator, destdir)


    def compute_morph(self, force_processing=False, save_hdd=None, save_minimal=True, destdir=None, 
                      show=False, verbose=False):
        if save_hdd is None:
            save_hdd = self.autosave
        if destdir is None:
            destdir = self.scan_folder_processed

        if not force_processing and self.exist("morph.pkl", destdir)[0]:
            if verbose:
                print(f"Morph already processed. Loading in progress...", end="")
            imported, is_data_from_disk = self.loadifexist("morph.pkl", destdir)
            if verbose:
                print("loaded.")
        else:
            if verbose:
                print(f"No processed morph found.")
            is_data_from_disk = False

        if is_data_from_disk:
            self.morph = imported
        else:
            if verbose:
                print(f"Processing morph...")
            if self.metadata.downsample:
                self.morph = OCTMorph(self.metadata, downsample_method=self.metadata.downsample_method)
            else:
                self.morph = OCTMorph(self.metadata)
            self.morph.compute_morph()

        if not save_hdd or is_data_from_disk:
            return
        if verbose:
            print(f"Saving in progress...\nLocation: {destdir})")

        if save_minimal:
            data = OCTMorph(self.metadata)
            data.morph = self.morph.morph
        else:
            data = self.morph
        self.saveVariable("morph.pkl", data, destdir)

        if show:
            path = os.path.join(destdir, ".png")
            self.morph.show(datatype="db", remove_initial_lines=True, save=True, save_path=path)


    def compute_phaseChange(self, force_processing=False, save_hdd=None, save_minimal=True, destdir=None, verbose=False):
        save_hdd = save_hdd if save_hdd is not None else self.autosave
        destdir = destdir if destdir is not None else self.scan_folder_processed

        if not force_processing and self.exist("phasechange.pkl", destdir)[0]:
            if verbose:
                print("Phase change already processed. Loading in progress...", end="")
            self.PChange, _ = self.loadifexist("phasechange.pkl", destdir)
            if verbose:
                print("loaded.")
        else:
            if verbose:
                print("No processed phase change found.")
            if self.metadata is None:
                self.load_metadata()
            if self.morph is None or self.morph.morph.shape[1] < 1024:
                self.compute_morph()
            self.PChange = OCTPhaseChange()
            self.PChange.compute_phase_change(self.morph.morph, self.metadata)

            if save_hdd:
                if verbose:
                    print(f"Saving in progress...\nLocation: {destdir}")
                matrix_save_path = os.path.join(destdir, 'morph_matrix.npy')
                try:
                    np.save(matrix_save_path, self.morph.morph)
                    print(f"Morph matrix data saved to {matrix_save_path}")
                except Exception as e:
                    print(f"Error saving morph matrix: {e}")
                self.saveVariable("phasechange.pkl", self.PChange, destdir)


    def loadifexist(self, datatype, destdir=None):
        exist, fullpath_fname = self.exist(datatype=datatype, destdir=destdir)
        if exist:
            with open(fullpath_fname, 'rb') as f:
                imported = pickle.load(f)
            data_already_exist = True
        else:
            imported = None
            data_already_exist = False

        return imported, data_already_exist
    

    def exist(self, datatype, destdir=None):
        if destdir is None:
            destdir = self.scan_folder_processed
        if not datatype.endswith('.pkl'):  # Change from '.pkl' to '.pkl'
            datatype += ".pkl"
        fullpath_fname = os.path.join(destdir, datatype)

        return os.path.isfile(fullpath_fname), fullpath_fname
    

    def saveVariable(self, datatype, data, destdir=None, verbose=False):
        if destdir is None:
            destdir = self.scan_folder_processed

        if not datatype.endswith('.pkl'):  # Change from '.pkl' to '.pkl'
            datatype += ".pkl"
        if data is None:
            raise ValueError(f"Something went wrong when trying to save the variable <{datatype}>.")
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        with open(os.path.join(destdir, datatype), 'wb') as f:
            pickle.dump(data, f)
        if verbose:
            print(f"{datatype} saved on HDD.")

    def find_fileabspath_from_node(self, target_name):
        filepath = self.scan_folder_raw
        while not any(target_name in name for name in os.listdir(filepath)):
            filepath, leaf_folder = os.path.split(filepath)
            if not leaf_folder:
                filepath = None
                break
        if leaf_folder:
            filepath = os.path.join(filepath, target_name)
        return filepath

    def fprintf(self, s):
        print(f"OCTRecordingManager :: {s}")

def movefile_local(sourcefile, destdir):
    try:
        shutil.move(sourcefile, destdir)
    except Exception as e:
        print(f'Failed to move file "{sourcefile}" to "{destdir}" because "{e}"')
        return
    print('file moved')