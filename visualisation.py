'''
Visualisation class that saves information to the brain data and a function to plot the accuracy of predictions.
Code from https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=S5-uT-S9zIQ0
'''
import numpy as np
import matplotlib.pyplot as plt
import os

class VisualizeFMRI:

    def __init__(self, data_dir):
        # Load the brain surface map of all vertices
        self.data_dir = data_dir
        roi_dir_left = os.path.join(data_dir, 'roi_masks/lh.all-vertices_fsaverage_space.npy')
        roi_dir_right = os.path.join(data_dir, 'roi_masks/rh.all-vertices_fsaverage_space.npy')
        self.fsaverage_all_vertices_left = np.load(roi_dir_left)
        self.fsaverage_all_vertices_right = np.load(roi_dir_right)
        self.rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    
    def define_roi_class(self, roi):
        # Define the ROI class based on the selected ROI
        if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = 'prf-visualrois'
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = 'floc-bodies'
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = 'floc-faces'
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_class = 'floc-places'
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = 'floc-words'
        elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
            roi_class = 'streams'
        return roi_class

    def visualize_rois(self, lh_correlation, rh_correlation, filename):
        # Load the ROI classes mapping dictionaries
        roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
            'mapping_floc-faces.npy', 'mapping_floc-places.npy',
            'mapping_floc-words.npy', 'mapping_streams.npy']
        roi_name_maps = []
        for r in roi_mapping_files:
            roi_name_maps.append(np.load(os.path.join(self.data_dir, 'roi_masks', r),
                allow_pickle=True).item())

        # Load the ROI brain surface maps
        lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
            'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
            'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
            'lh.streams_challenge_space.npy']
        rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
            'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
            'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
            'rh.streams_challenge_space.npy']
        lh_challenge_rois = []
        rh_challenge_rois = []
        for r in range(len(lh_challenge_roi_files)):
            lh_challenge_rois.append(np.load(os.path.join(self.data_dir, 'roi_masks',
                lh_challenge_roi_files[r])))
            rh_challenge_rois.append(np.load(os.path.join(self.data_dir, 'roi_masks',
                rh_challenge_roi_files[r])))

        # Select the correlation results vertices of each ROI
        roi_names = []
        lh_roi_correlation = []
        rh_roi_correlation = []
        for r1 in range(len(lh_challenge_rois)):
            for r2 in roi_name_maps[r1].items():
                if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                    roi_names.append(r2[1])
                    lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                    rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                    lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                    rh_roi_correlation.append(rh_correlation[rh_roi_idx])
        roi_names.append('All vertices')
        lh_roi_correlation.append(lh_correlation)
        rh_roi_correlation.append(rh_correlation)

        # Create the plot
        lh_median_roi_correlation = [np.median(lh_roi_correlation[r])
            for r in range(len(lh_roi_correlation))]
        rh_median_roi_correlation = [np.median(rh_roi_correlation[r])
            for r in range(len(rh_roi_correlation))]
        plt.figure(figsize=(18,6))
        x = np.arange(len(roi_names))
        width = 0.30
        plt.bar(x - width/2, lh_median_roi_correlation, width, label='Left Hemisphere')
        plt.bar(x + width/2, rh_median_roi_correlation, width,
            label='Right Hemishpere')
        plt.xlim(left=min(x)-.5, right=max(x)+.5)
        plt.ylim(bottom=0, top=1)
        plt.xlabel('ROIs')
        plt.xticks(ticks=x, labels=roi_names, rotation=60)
        plt.ylabel('Median Pearson\'s $r$')
        plt.legend(frameon=True, loc=1)
        plt.savefig(filename)
        plt.show()