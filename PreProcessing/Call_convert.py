import os
import Convert2Nifti
args = ['/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/Images_png_'+f'{i:02d}'+'/Images_png' for i in range(1,57)]
for arg in args:
    Convert2Nifti.main(arg)