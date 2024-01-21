import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

'''=========================Confusion Matrix Visualization========================='''
# SVM
cm1 = scipy.io.loadmat("E:/11.12/cm(SVM).mat")
cm1 = cm1['cf_mt']
cm1 = cm1[0:5, 0:5]

# ELM
cm2 = scipy.io.loadmat("E:/11.12/cm(ELM).mat")
cm2 = cm2['cf_mt']
cm2 = cm2[0:5, 0:5]

# 1DCNN
cm3 = scipy.io.loadmat("E:/11.12/cm(1DCNN).mat")
cm3 = cm3['cm_all']
cm3 = cm3[0:5, 0:5]

# 1DCNN+CAM
cm4 = scipy.io.loadmat("E:/11.12/cm(1DCNN+CAM).mat")
cm4 = cm4['cm_all']
cm4 = cm4[0:5, 0:5]


con_mat_norm = cm4.astype('float') / cm4.sum(axis=1)[:, np.newaxis]  # normalize
con_mat_norm = np.around(con_mat_norm, decimals=4)
# List of category names
class_names = ['1', '2', '3', '4', '5']
# === plot ===
plt.figure()
sns.heatmap(con_mat_norm, annot=True, fmt='.1%', cmap='GnBu', xticklabels=class_names, yticklabels=class_names)
# Adjust the x-axis label direction to horizontal
plt.xticks(rotation=0)
plt.xlabel('Predicted label', fontsize=13)
plt.ylabel('True label', fontsize=13)
plt.savefig("1DCNN+CAM.tif", dpi=300, bbox_inches='tight')
plt.show()
