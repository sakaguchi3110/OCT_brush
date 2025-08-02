import matplotlib.pyplot as plt
import numpy as np


class OCTPhaseChange:
    def __init__(self):
        self.phase_change = None  # phase variation of the morph data

    def compute_phase_change(self, morph, metadata, show=False):
        self.session_id = metadata.session_name
        nLines, nDepths, nSamples = morph.shape
        phase_change_ = np.zeros((nLines, nDepths, nSamples - 1))
        
        for a in range(nLines):
            print(f"Phase Change processing ({a + 1}/{nLines})")
            phase_change_[a, :, :] = self.compute_phase_change_slice(morph[a, :, :])
        self.phase_change = phase_change_
        
        if show:
            for a in range(nLines):
                fig, axs = plt.subplots(3, 1, figsize=(10, 15))

                im = axs[0].imshow(np.abs(morph[a, :, :]), cmap='gray', aspect='auto')
                axs[0].set_title('Amplitude Morph Image')
                axs[0].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[0])

                im = axs[1].imshow(np.angle(morph[a, :, :]), cmap='gray', aspect='auto')
                axs[1].set_title('Phase Morph Image')
                axs[1].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[1])

                im = axs[2].imshow(self.phase_change[a, :, :], cmap='gray', aspect='auto')
                axs[2].set_title('Slice Phase Change Image')
                axs[2].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[2])

                # Add main title
                fig.suptitle(f'{self.session_id}, a-line={a+1}/{nLines}', fontsize=16)
                axs[2].set_xlabel('Time (nsample)')

                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
                
                # Maximize the figure window
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                plt.show(block=True)

        print("Phase Change is done.")


    def compute_phase_change_slice(self, slice_morph, old=False):
        slice_morph = np.squeeze(slice_morph)
        if old:
            ntimes = slice_morph.shape[1]
            phasechange = np.angle(slice_morph[:, 1:] * np.conj(slice_morph[:, :-1]))
        else:
            phasechange = np.diff(np.angle(slice_morph), axis=1)
            phasechange[phasechange > +np.pi] -= 2 * np.pi
            phasechange[phasechange < -np.pi] += 2 * np.pi
        return phasechange
    
def get_phase_change_data(octr):
    phase_change_data = octr.phase_change  
    return phase_change_data
