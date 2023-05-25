title: "Connectivity Circular Plot"
linkTitle: "Example Code"
weight: 1
tags: ["Connectivity", "MEG", "MNE python", "freesurfer", "Coregistration"]
author: Benjamin M Slade and Will Woods
description: > This tutorial will produce one circular connectivity plot from epochs. 
Email: bslade@swin.edu.au, wwoods@swin.edu.au
Github: @benmslade
Twitter: @Benmslade
 
To generate a connectivity plot, the forward model and source space need to be computer. To generate thos files the Boundry Element Models and the files needed co-registration of MEG and MRI data have to be created from the raw DICOM files. 

To do this, run the script below: XXX is the project number.
```
python /dagg/public/neuro/freesurfer_MEG_scripts/do_all_freesurfer.py ozXXX 
```
The contents of this script are below
```
#!/bin/bash

export FREESURFER_HOME=/dagg/public/neuro/freesurfer-linux-centos7_x86_64-7.1.1
export SUBJECTS_DIR=/fred/oz120/freesurfer/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /fred/oz120/freesurfer/scripts

SUBJECT=$2

echo $SUBJECT
echo "$@" -qcache -3T

echo
echo XXXXXXXXXXXXXXXXXX
echo Starting recon-all
echo

#recon-all "$@" -qcache -3T
recon-all "$@" -qcache -3T

echo
echo Finished recon-all
echo XXXXXXXXXXXXXXXXXX
echo

#----- Do alternative brain extractions, ANTS and HD-BET -------#

module load apptainer/latest

# Make a reoriented copy of the T1W image, required for HD-BET
# Create a new subdirectory in the freesurfer folder to hold the output of both HD-BET and ANTS brain extraction

BRAIN_EXT_DIR=$SUBJECTS_DIR/$SUBJECT/alt_brain_ext

mkdir $BRAIN_EXT_DIR

T1_orig_001=$SUBJECTS_DIR/$SUBJECT/mri/orig/001.mgz

# Used in HD-BET
mri_convert --in_type mgz --out_type nii $T1_orig_001 $BRAIN_EXT_DIR/001.nii.gz
apptainer exec  --bind /fred,/dagg,/home /dagg/public/neuro/containers/fmriprep-1.5.4.simg fslreorient2std $BRAIN_EXT_DIR/001.nii.gz $BRAIN_EXT_DIR/T1w_reoriented.nii.gz

# Do all the ANTS coregistration, and both alternate brain extractions using ANTS and HD-BET

apptainer exec  --bind /fred,/dagg,/home --nv /dagg/public/neuro/cuda_ants_28_08_2020.sif /fred/oz120/freesurfer/scripts/MEGMRI_preproc.sh $SUBJECT
(END)
---
```
Currently on OzStar, this container is needed when running connectivity analysis. 
ml apptainer/latest
singularity shell --bind /fred,/dagg/public/neuro  /dagg/public/neuro/containers/mneextended_1.2.2_20221207.sif
source /opt/miniconda-4.7.12/etc/profile.d/conda.sh
conda activate mne-extended

Installations:
Auto_Reject needs to be installed: e.g., pip install -U autoreject. Available here: (https://autoreject.github.io/stable/index.html)
mne_connectivity is a seperate package and requires installation: e.g., pip install mne_connectivity. Available here: (https://mne.tools/mne-connectivity/stable/index.html)

```
import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys
sys.path = ['/home/bslade/pytmp'] + sys.path #this paths to where the mne_connectivity is saved too
import mne_connectivity

import mne
#from autoreject import AutoReject #if using autoreject to detemine rejection threshold uncomment this line and the next. 
#from autoreject import get_rejection_threshold
#reject = get_rejection_threshold(epochs)
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs, corrmap
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, read_inverse_operator
from mne.datasets import sample
from mne import setup_volume_source_space, setup_source_space
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
```
**Pre-Processing Raw MEG Data**

Import raw file
```
raw_fname = '/home/bslade/AEDAPT/MI02-sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg.fif'
raw = mne.io.read_raw_fif(raw_fname, preload = True, verbose = False)
raw.info['bads'] = ['MEG2131','MEG0143']
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
raw.load_data
#Band pass filter from 1 - 40 Hz.
raw.filter(1., 40., fir_design= 'firwin')
#View the data to check the filter applied
raw.plot(group_by='selection')
```
#To generate the transform file needed to perform co-registration the -trans.fif file needs to be loaded. 
#to generate the -trans.fif files, in the terminal copy and paste: 
```
mne coreg --subjects=/fred/oz120/freesurfer/subjects --high-res-head
```
#Instructions on how to use the mne coreg are here: (https://mne.tools/1.1/auto_tutorials/forward/20_source_alignment.html)
#Readin the saved -trans.fif file. The -trans.fif file is needed to produce the forward solution and source space file. 
```
trans = '/home/bslade/AEDAPT/MI02-sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-trans.fif'
```
ICA analysis - removes artifact generated from eye movements and the heart. 
```
raw.load_data()
ica = ICA(n_components=0.95, method='fastica')
ica.fit(raw, decim = 30)
ica
ica.plot_components()

#Remove EOG
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='BIO001')
ica.exclude = eog_indices
ica.plot_scores(eog_scores) 

#Remove EEC
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name= 'BIO002')
ica.exclude = ecg_indices
ica.plot_scores(ecg_scores)

ica.exclude.extend(eog_indices) #ICA will now reject this component
ica.exclude.extend(ecg_indices)

ica.apply(raw)

```
Create events from resting state data
```
tmin,tmax = 0,3   #3 second epochs
baseline = None
event_id = 1
events = mne.event.make_fixed_length_events(raw, event_id, duration=tmax-tmin)
```

A rejection theshold is needed when generating epochs to reject noisy epochs. For this tutorial, the rejection theshold is set by Auto_reject, but can be determined manually. 
```
reject = dict(mag=1.96e-11, grad=3.50e-10) #This rejection threshold was selected for this raw.fif file. 
#reject = get_rejection_threshold(epochs) #Uncomment this if Auto_reject was used to determine the rejection threshold for noisy epochs. 
```
Creating epochs
```
epochs = mne.Epochs(raw, events, baseline=(0.0, None), tmin=tmin, tmax=tmax, event_id=event_id, picks=picks, reject=reject, preload=True)
#epochs_clean = ar.fit_transform(epochs) 
#Epochs can be saved to be loaded at another time using the code examples below:
#epochs.save('.../MI02-sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-epo.fif', overwrite=True) #Use overwrite=True to save over files
#Read Epochs back, checking the above file saved
#mne.read_epochs('.../MI02-sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-epo.fif', preload=True)
```

Create the covariance matrix from the emptyroom recording. The Covariance matrix can be create from the epochs if empty room recordings do not exist by using (https://mne.tools/stable/generated/mne.compute_covariance.html)
```
emptyroom = '/home/bslade/AEDAPT/MI02-sub-TEST/ses-TEST/meg/sub-TEST_ses-emptyroom_task-emptyroom_meg.fif'
raw_emptyroom = mne.io.read_raw_fif(emptyroom, preload = True, verbose = False)
noise_cov = mne.compute_raw_covariance(raw_emptyroom)

#The covariance matrix can be save using the code examples below:
#noise_cov.save(.../MI02-sub-TESTsub-TEST/ses-TEST/meg/sub-TEST_ses-emptyroom_task-emptyroom_meg-cov.fif', overwrite=False) #Use overwrite=True to save over files. 
#Read covariance matrix back, checking the above file saved
#noise_cov = mne.read_cov(.../MI02-sub-TESTsub-TEST/ses-TEST/meg/sub-TEST_ses-emptyroom_task-emptyroom_meg-cov.fif')
```

**Generating the connectivity circle Plot**

#Set the directory to the MRI files and scan date/participant
```
subjects_dir = '/fred/oz120/freesurfer/subjects/'
subject = 'MI02-sub-TEST'
```
Is wanted, a list of sub structures can be shown on the connectivity circle plot byt selecteing the sub strucutres to include in the source space"
```
labels_vol = ['Left-Amygdala',
              'Left-Thalamus-Proper',
              'Left-Cerebellum-Cortex',
              'Brain-Stem',
              'Right-Amygdala',
              'Right-Thalamus-Proper',
              'Right-Cerebellum-Cortex']
```
Setup a surface-based source space. The parameteres and options used here are recommended by MNE Python when conducting MEG analysis. 
```
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
print(src)
sphere = (0.0, 0.0, 0.04, 0.09)
# Setup a volume source space
vol_src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir, pos=5.0, add_interpolator=True, volume_label=labels_vol, sphere=sphere)
print(vol_src)
# Generate the mixed source space
src += vol_src

scan_date = 'MI02-sub-TEST'   #Change this and 'subjects_dir' for other subjects
bem_sol_fn = os.path.join(subjects_dir, scan_date, 'bem', '%s-5120-bem-sol.fif'%scan_date)
bem = mne.read_bem_solution(bem_sol_fn)
#Generate the forward soution
fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=2)
print(fwd)
#Can save these files using the code examples below, but not nessessary:
#mne.write_source_spaces(.../sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-src.fif', src, overwrite=True, verbose=None)
#mne.write_forward_solution(.../sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-fwd.fif', fwd, overwrite=True, verbose=None)

#Read source space and forward solution .fif files
#scr = mne.read_source_spaces(.../sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-src.fif', patch_stats=False, verbose=None)
#fwd = mne.read_forward_solution(.../sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-fwd.fif', include=(), exclude=('bads'),verbose=None)

snr = 1.0           
inv_method = 'dSPM' # Can use MNE or sLORETA
parc = 'aparc'      # The specifiies which parcellation (atlas) to use. Can use 'aparc' 'aparc.a2009s' instead. 
lambda2 = 1.0 / snr ** 2

# Compute inverse operator
inv_operator = make_inverse_operator(epochs.info, fwd, noise_cov, depth=None, fixed=False)
#Can save the inverse operator using the code exmaples below:
#inv = mne.minimum_norm.write_inverse_operator('.../sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-inv.fif', inv_operator, overwrite=True, verbose=None)
#Read the /-inv.fif files. 
#inv = mne.minimum_norm.read_inverse_operator('.../sub-TEST/ses-TEST/meg/sub-TEST_ses-rest_task-rest_meg-inv.fif')

stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, inv_method,pick_ori=None, return_generator=True)
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels per a hemisphere
labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)

# Average the source estimates within each label of the cortical parcellation and each sub-structure contained in the source space.
# When mode = 'mean_flip', this option is used only for the cortical labels.
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(
    stcs, labels_parc, src, mode='mean_flip', allow_empty=True,
    return_generator=True)

#Compute the connectivity (for the desired frequency band:delta:1Hz - 4Hz, theta:4Hz-8Hz, alpha:8Hz-13Hz, beta:13Hz-30Hz, gamma:30-Hz-50Hz)and plot it using a circular graph layout
fmin = 8. 
fmax = 13.
sfreq = epochs.info['sfreq']  # The sampling frequency
con = spectral_connectivity_epochs(
    label_ts, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

#Create a list of Label containing also the sub structures
labels_aseg = mne.get_volume_labels_from_src(src, subject, subjects_dir)
labels = labels_parc + labels_aseg

# Read colors
node_colors = [label.color for label in labels]

#Reorder labels based on their location in the left hemisphere
label_names = [label.name for label in labels]
lh_labels = [name for name in label_names if name.endswith('lh')]
rh_labels = [name for name in label_names if name.endswith('rh')]

# Get the y-location of the label
label_ypos_lh = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos_lh.append(ypos)
try:
    idx = label_names.index('Brain-Stem')
except ValueError:
    pass
else:
    ypos = np.mean(labels[idx].pos[:, 1])
    lh_labels.append('Brain-Stem')
    label_ypos_lh.append(ypos)


# Reorder labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

# For right hemisphere
rh_labels = [label[:-2] + 'rh' for label in lh_labels
             if label != 'Brain-Stem' and label[:-2] + 'rh' in rh_labels]

# Save the plot order
node_order = lh_labels[::-1] + rh_labels

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) // 2])


# Plot the graph using node colors from the FreeSurfer parcellation. By default, MNE Python only shows 300 strongest connections.
conmat = con.get_data(output='dense')[:, :, 0]
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
plot_connectivity_circle(conmat, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=node_colors,
                         title='All-to-All Connectivity left-Auditory '
                         'Condition (PLI)', ax=ax)
fig.tight_layout()
```
 **Saving Images to file**
 
 The generated connectivity plot can be saved as a high res (300dpi) image for publiction using the example code below. 
 
 ```
 plt.rcParams['savefig.facecolor']='black'
 plt.savefig('../.png', dpi = 300, edgecolor='none') #edgecolour=None ensure the plot can be seen from the background
 ```
 
