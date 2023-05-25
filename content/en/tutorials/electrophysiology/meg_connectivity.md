This tutorial will produce circular connectivity plots
 
Firstly, the Boundry Element Models and the files needed co-registration of MEG and MRI data have to be created from the raw DICOM files. 
To do this, run:
```
python /fred/oz120/freesurfer/scripts/do_freesurfer_oz120.sh
```
This script will run
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

#Currently on OzStar, this container is needed when running connectivity analysis. 
#ml apptainer/latest
#singularity shell --bind /home/,/fred,/dagg/public/neuro  /dagg/public/neuro/containers/will/cuda_ants_cupy_10.2_Py3.9_R_v2b.sif

import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import glob

import mne
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
#mne_connectivity is a seperate package and requires installation (e.g., pip install mne_connectivity)

#ozstar_project = 'XXX'
#meg_path = '/fred/%s/raw/meg/'%ozstar_project

#file_list = list(Path(meg_path).rglob('*.fif'))
#for count,path in enumerate(file_list):
    #raw_fname = os.path.join(str(path.parent),path.name)
    #raw = mne.io.read_raw_fif(raw_fname, preload = True, verbose = False)
    #print(raw.info)
    #print ('\n%d/%d %s\n'%(count+1,len(file_list),raw))


#Import raw file
raw_fname = r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-rest_task-rest_meg.fif'
raw = mne.io.read_raw_fif(raw_fname, preload = True, verbose = False)
print(raw.info)



#To generate the trans file co-registration is needed before inporting the -trans.fif file
#In the terminal use:
#singularity shell --bind /fred,/dagg/public/neuro  /dagg/public/neuro/containers/mneextended_1.2.2_20221207.sif
#source /opt/miniconda-4.7.12/etc/profile.d/conda.sh
#conda activate mne-extended
#mne coreg --subjects=/fred/oz80/freesurfer/subjects --high-res-head

#Instructions on how to use the mne coreg are here: https://mne.tools/1.1/auto_tutorials/forward/20_source_alignment.html

trans = r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-rest_task-rest_meg-trans.fif' #import -trans.fif file. Needed for forward solution and src files. 
#src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
#print(src)
#sphere = (0.0, 0.0, 0.04, 0.09)
#vol_src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir, sphere=sphere) #sphere_units='m')
#print(vol_src)

#subjects_dir = 'fred/oz/freesurfer/subjects/'
#subject = '2213xTOUCH_001'

#scan_date = '2213xTOUCH_001'   #Change this and 'subjects_dir' for other subjects

#bem_sol_fn = os.path.join(subjects_dir, scan_date, 'bem', '%s-5120-bem-sol.fif'%scan_date)

#bem = mne.read_bem_solution(bem_sol_fn)
#fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=2)
#print(fwd)

#Write source space and forward solution files
#mne.write_source_spaces(r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-rest_task-rest_meg-src.fif', src, overwrite=True, verbose=None)
#mne.write_forward_solution(r'/C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-rest_task-rest_meg-fwd.fif', fwd, overwrite=True, verbose=None)

#Read source space and forward solution files
#mne.read_source_spaces(r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-rest_task-rest_meg-src.fif', patch_stats=False, verbose=None)
#mne.read_forward_solution(r'/C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-rest_task-rest_meg-fwd.fif', include=(), exclude=('bads'),verbose=None)

#Creating the forward solution. The source space needs to the computed before the forward solution is computed.

raw.info['bads'] = ['MEG2131','MEG0143']
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
raw.load_data
raw.filter(1., 40., fir_design= 'firwin')
raw.plot(group_by='selection')

reject = dict(mag=1.96e-11, grad=3.50e-10)

#ICA analysis - removes activity generated from eye movements and the heart. 
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


#Create events
tmin,tmax = 0,3   #3 second epochs
baseline = None
event_id = 1
events = mne.event.make_fixed_length_events(raw, event_id, duration=tmax-tmin)

#Create epochs
epochs = mne.Epochs(raw, events, baseline=(0.0, None), tmin=0.0, tmax=3, event_id=event_id, picks=picks, reject=reject, preload=True)
#Save epochs
epochs.save(r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-emptyroom_task-emptyroom_meg-epo.fif', overwrite=True)
#Read Epochs back, checking the above file saved
mne.read_epochs(r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-emptyroom_task-emptyroom_meg-epo.fif', preload=True)

#
#dont haveto save these, can just generate these withouth saving
#Create the covariance matrix from the emptyroom recording
emptyroom = r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-emptyroom_task-emptyroom_meg.fif'
raw_emptyroom = mne.io.read_raw_fif(emptyroom, preload = True, verbose = False)
noise_cov = mne.compute_raw_covariance(raw_emptyroom)
#Save covariance matrix
noise_cov.save(r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-emptyroom_task-emptyroom_meg-cov.fif', overwrite=True)
#Read covariance matrix back, checking the above file saved
noise_cov = mne.read_cov(r'C:\Users\bslade\Desktop\sub-TEST\ses-TEST\meg\sub-TEST_ses-emptyroom_task-emptyroom_meg-cov.fif')


#Connectivity - Circle Plot
data_path = sample.data_path()
subject = 'sample'
data_dir = op.join(data_path, 'MEG', subject)
subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')

# Set file names
fname_aseg = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')

fname_model = op.join(bem_dir, '%s-5120-bem.fif' % subject)
fname_bem = op.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
fname_trans = op.join(data_dir, 'sample_audvis_raw-trans.fif')

# List of sub structures we are interested in. We select only the
# sub structures we want to include in the source space
labels_vol = ['Left-Amygdala',
              'Left-Thalamus-Proper',
              'Left-Cerebellum-Cortex',
              'Brain-Stem',
              'Right-Amygdala',
              'Right-Thalamus-Proper',
              'Right-Cerebellum-Cortex']

# Setup a surface-based source space, oct5 is not very dense (just used
# to speed up this example; we recommend oct6 in actual analyses)
src = setup_source_space(subject, subjects_dir=subjects_dir,
                         spacing='oct5', add_dist=False)
# Setup a volume source space
# set pos=10.0 for speed, not very accurate; MNE Python recommend something smaller
# like 5.0 in actual analyses:
vol_src = setup_volume_source_space(
    subject, mri=fname_aseg, pos=10.0, bem=fname_model,
    add_interpolator=False,  # just for speed, usually use True
    volume_label=labels_vol, subjects_dir=subjects_dir)
# Generate the mixed source space
src += vol_src

fwd = make_forward_solution(raw.info, fname_trans, src, fname_bem,
                            mindist=5.0) 

snr = 1.0           # use smaller SNR for raw data
inv_method = 'dSPM'
parc = 'aparc'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'

lambda2 = 1.0 / snr ** 2

# Compute inverse operator
inverse_operator = make_inverse_operator(
    epochs.info, fwd, noise_cov, depth=None, fixed=False)
del fwd

stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, inv_method,
                            pick_ori=None, return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels_parc = mne.read_labels_from_annot(subject, parc=parc,
                                         subjects_dir=subjects_dir)

# Average the source estimates within each label of the cortical parcellation
# and each sub-structure contained in the source space.
# When mode = 'mean_flip', this option is used only for the cortical labels.
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(
    stcs, labels_parc, src, mode='mean_flip', allow_empty=True,
    return_generator=True)

# We compute the connectivity in the alpha band and plot it using a circular
# graph layout
fmin = 8. #Can change the frequency bands here (delta:1Hz - 4Hz, theta:4Hz-8Hz, alpha:8Hz-13Hz, beta:13Hz-30Hz, gamma:30-Hz-50Hz)
fmax = 13.
sfreq = epochs.info['sfreq']  # the sampling frequency
con = spectral_connectivity_epochs(
    label_ts, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

# We create a list of Label containing also the sub structures
labels_aseg = mne.get_volume_labels_from_src(src, subject, subjects_dir)
labels = labels_parc + labels_aseg

# read colors
node_colors = [label.color for label in labels]

# We reorder the labels based on their location in the left hemi
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


# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels
             if label != 'Brain-Stem' and label[:-2] + 'rh' in rh_labels]

# Save the plot order
node_order = lh_labels[::-1] + rh_labels

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) // 2])


# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
conmat = con.get_data(output='dense')[:, :, 0]
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
plot_connectivity_circle(conmat, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=node_colors,
                         title='All-to-All Connectivity left-Auditory '
                         'Condition (PLI)', ax=ax)
fig.tight_layout()


title: "Template for tutorial creation"
linkTitle: "Workflow template"
weight: 1
tags: ["template", "documentation"]
author: Angela I. Renton
description: > 
  Follow this template to contribute your own tutorial to the Neurodesk documentation.
---
<!--
Begin setting up your tutorial by filling in the details in the description above. This controls how your tutorial is named and displayed in the Neurodesk documentation. The details are as follows:

title: A title for your tutorial
linkTitle: A shortened version of the title for the menu
weight: This controls where in the menu your tutorial will appear; you can leave this set to 1 for default sorting 
tags: List any number of tags to help others find this tutorial. i.e. "eeg", "mvpa", "statistics"
description: > a short description of your tutorial. This will form the subheading for the tutorial page. 

Once you've filled out those details, you can delete this comment block. 
-->

> _This tutorial was created by Name P. Namington._ 
>
> Email: n.namington@institution.edu.au
>
> Github: @Namesgit
>
> Twitter: @Nameshandle
>
<!-- Fill in your personal details above so that we can credit the tutorial to you. Feel free to add any additional contact details i.e. website, or remove those that are irrelevant -->

Welcome to the workflow (tutorial) template, which you can use to contribute your own neurodesk workflow to our documentation. We aim to collect a wide variety of workflows representing the spectrum of tools available under the neurodesk architecture and the diversity in how researchers might apply them. Please add plenty of descriptive detail and make sure that all steps of the workflow work before submitting the tutorial. 

## How to contribute a new workflow

Begin by creating a copy of our documentation that you can edit:
1. Visit the github repository for the Neurodesk documentation (https://github.com/NeuroDesk/neurodesk.github.io).
2. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository.
- _You should now have your own copy of the documentation, which you can alter without affecting our official documentation. You should see a panel stating "This branch is up to date with Neurodesk:main." If someone else makes a change to the official documentation, the statement will change to reflect this. You can bring your repository up to date by clicking "Fetch upstream"._ 

Next, create your workflow:
1. [Clone](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository) your forked version of our documentation to a location of your choice on your computer. 
2. In this new folder, navigate to "neurodesk.github.io/content/en/tutorials" and then navigate to the subfolder you believe your workflow belongs in (e.g. "/functional_imaging"). 
3. Create a new, appropriately named markdown file to house your workflow. (e.g. "/physio.md")
4. Open this file in the editor of your choice (we recommend [vscode](https://code.visualstudio.com/)) and populate it with your workflow! Please use this template as a style guide, it can be located at "neurodesk.github.io\content\en\tutorials\documentation\workflowtemplate.md". You're also welcome to have a look at other the workflows already documented on our website for inspiration. 

Finally, contribute your new workflow to the official documentation!:
1. Once you are happy with your workflow, make sure you [commit](https://github.com/git-guides/git-commit) all your changes and [push](https://github.com/git-guides/git-push) these local commits to github.
2. Navigate to your forked version of the repository on github.
3. Before you proceed, make sure you are up to date with our upstream documentation, you may need to [fetch upstream changes](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).
4. Now you can preview the changes before contributing them upstream. For this click on the "Actions" tab and enable the Actions ("I understand my workflows..."). The first build will fail (due to a bug with the Github token), but the second build will work.
5. Then you need to open the settings of the repository and check that Pages points to gh-pages and when clicking on the link the site should be there.
6. To contribute your changes, click "Contribute", and then ["Open pull request"](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).
7. Give your pull request a title (e.g. "Document PhysIO workflow"), leave a comment briefly describing what you have done, and then create the pull request. 
8. Someone from the Neurodesk team will review and accept your workflow and it will appear on our website soon!. 

Thanks so much for taking the time to contribute your workflow to the Neurodesk community! If you have any feedback on the process, please let us know on [github discussions](https://github.com/orgs/NeuroDesk/discussions).

## Formatting guidelines

You can embelish your text in this tutorial using markdown conventions; text can be **bold**, _italic_, or ~~strikethrough~~. You can also add [Links](https://www.neurodesk.org/), and you can organise your tutorial with headers, starting at level 2 (the page title is a level 1 header):

## Level 2 heading

You can also include progressively smaller subheadings:

### Level 3 heading

Some more detailed information. 

#### Level 4 heading

Even more detailed information. 

### Code blocks

You can add codeblocks to your tutorial as follows:

```none
# Some example code
import numpy as np
a = np.array([1, 2])
b = np.array([3, 4])
print(a+b)
```

Or add syntax highlighting to your codeblocks:
```go
# Some example code
import numpy as np
a = np.array([1, 2])
b = np.array([3, 4])
print(a+b)
```

Advanced code or command line formatting using this html snippet:
```bash
# Some example code
import numpy as np
a = np.array([1, 2])
b = np.array([3, 4])
print(a+b)
[4 6]
```

You can also add code snippets, e.g. `var foo = "bar";`, which will be shown inline.

### Images

To add screenshots to your tutorial, create a subfolder in `neurodesk.github.io/static` with the same link name as your tutorial. Add your screenshot to this folder, keeping in mind that you may want to adjust your screenshot to a reasonable size before uploading. You can then embed these images in your tutorial using the following convention: 

```
![EEGtut1](/EEG_Tutorial/EEGtut1.png 'EEGtut1') <!-- ![filename without extension](/subfolder_name/filename.png '[filename without extension')  -->
```
![EEGtut1](/EEG_Tutorial/EEGtut1.png 'EEGtut1') <!-- ![filename without extension](/subfolder_name/filename.png '[filename without extension')  -->

### Alerts and warnings

You can grab reader's attention to particularly important information with quoteblocks, alerts and warnings:

> This is a quoteblock

{{< alert >}}This is an alert.{{< /alert >}}
{{< alert title="Note" >}}This is an alert with a title.{{< /alert >}}
{{< alert color="warning" >}}This is a warning.{{< /alert >}}
{{< alert color="warning" title="Warning" >}}This is a warning with a title.{{< /alert >}}

You can also segment information as follows:

----------------

There's a horizontal rule above and below this.

----------------

Or add page information:
{{% pageinfo %}}
This is a placeholder. Replace it with your own content.
{{% /pageinfo %}}

### Tables

You may want to order information in a table as follows:

| Neuroscientist           | Notable work                                         | Lifetime  |
|--------------------------|------------------------------------------------------|-----------|
| Santiago Ramón y Cajal   | Investigations on microscopic structure of the brain | 1852–1934 |
| Rita Levi-Montalcini     | Discovery of nerve growth factor (NGF)               | 1909–2012 |
| Anne Treisman            | Feature integration theory of attention              | 1935–2018 |

### Lists

You may want to organise information in a list as follows:

Here is an unordered list:

* Rstudio
* JASP
* SPSS

And an ordered list:

1. Collect data
2. Try to install analysis software
3. Cry a little

And an unordered task list:

- [x] Install Neurodesktop
- [x] Analyse data
- [ ] Take a vacation

And a "mixed" task list:

- [ ] writing
- ?
- [ ] more writing probably

And a nested list:

* EEG file extensions
  * .eeg, .vhdr, .vmrk
  * .edf
  * .bdf
  * .set, .fdt
  * .smr
* MEG file extensions
  * .ds
  * .fif
  * .sqd
  * .raw
  * .kdf
