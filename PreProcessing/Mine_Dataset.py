import numpy as np
import os

def mine_folder(dir_add):
    files = sorted(os.listdir(dir_add))
    print(files[:10])
    patients = [x[:6] for x in files]   # number of patients in the dataset
    print('Number of patients in this folder is: ' + str(len(set(patients))))
    studies = [x[:9] for x in files]    # number of studies in the dataset; Each patient often underwent multiple CT examinations (studies) for different purposes or follow-up. i.e., this could be different followups
    print('Number of studies in this folder is: ' + str(len(set(studies))))
    series = [x[:12] for x in files] # Each study contains multiple volumes (series) that are scanned at the same time point but differ in image filters, contrast phases, etc.
    print('Number of series in this folder is: ' + str(len(set(series))))

if __name__ == "__main__":
    mine_folder(dir_add = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/nifti_dataset')