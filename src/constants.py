"""From assigment description:
background (label 0), necrotic and non-enhancing tumor (label 1), peritumoral edema (label 2) and GD-enhancing
tumor (label 4).

From JSON:
"labels": { 
    "0": "background", 
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour"
}, 
"""

# Image types in each 4D image
FLAIR = 0
T1_W = 1
T1_GD = 2
T2_W = 3

# Mask types
BG = 0
EDEMA = 1
NON_ENHANCING = 2
ENHANCING = 3

IMG_TYPES = ['FLAIR', 'T1_W', 'T1_GD', 'T2_W']
MASK_TYPES = ['BG', 'EDEMA', 'NON_ENHANCING', 'ENHANCING']

""""From assigment description:
The segmentation accuracy is measured by the Dice score and the Hausdorff distance (95%) metrics
for enhancing tumor region (ET, label 4), regions of the tumor core (TC, labels 1 and 4), and the
whole tumor region (WT, labels 1, 2 and 4).

No label 4, so we use 3 instead which corresponds to enhancing tumor.
"""
# WT = [EDEMA, NON_ENHANCING, ENHANCING]
WT = 0
# TC = [NON_ENHANCING, ENHANCING]
TC = 1
# ET = [ENHANCING]
ET = 2

AGGREGATE_MASK_TYPES = ['Whole tumor', 'Tumor core', 'Enhancing tumor']
