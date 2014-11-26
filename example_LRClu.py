"""
LRClu
"""

import nibabel
from nilearn import datasets


dataset = datasets.fetch_adhd()
img=nibabel.load(dataset.func[0])
X=img.get_data()
x,y,z,t = X.shape




# Author: Alexis Benichoux