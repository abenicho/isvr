# -*- coding: utf-8 -*-
from nilearn import datasets

dataset = datasets.fetch_adhd()
func_files = dataset.func # The list of 4D nifti files for each subject

### Apply CanICA ##############################################################
from nilearn.decomposition.canica import CanICA

n_components = 20
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_files)

# Retrieve the independent components in brain space
components_img = canica.masker_.inverse_transform(canica.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename('canica_resting_state.nii.gz')

### Visualize the results #####################################################
# Show some interesting components
import nibabel
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

for i in range(n_components):
    plot_stat_map(nibabel.Nifti1Image(components_img.get_data()[..., i],
                                      components_img.get_affine()),
                  display_mode="z", title="IC %d"%i, cut_coords=1,
                  colorbar=False)

plt.show()
