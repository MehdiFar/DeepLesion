import numpy as np
import pandas as pd
import visualize_image_mask as vim
import os
import nibabel as nib

def convert2NumPy(csv_dir, nifti_dir, save_dir):
    df = pd.read_csv(csv_dir)
    counter = 0
    for idx, row in df.iterrows():
        volume_name = "_".join(row['File_name'].split("_")[:-1])
        slice_index = row['Key_slice_index']
        bbox = row['Bounding_boxes']
        lesion_type = row['Coarse_lesion_type']
        Possibly_noisy = row['Possibly_noisy']
        Spacing_mm_px = row['Spacing_mm_px_']
        Image_size = row['Image_size']
        Train_Val_Test = row['Train_Val_Test']
        slice_start, slice_end = row['Slice_range'].split(', ')[0], row['Slice_range'].split(', ')[1]
        if Possibly_noisy == 1: 
            continue
        if Image_size.split(',')[0] != '512':
            continue

        print(idx)
        print(volume_name)

        if Train_Val_Test == 1:
            file_save_dir = os.path.join(save_dir, 'Train')
        if Train_Val_Test == 2:
            file_save_dir = os.path.join(save_dir, 'Tune')
        if Train_Val_Test == 3:
            file_save_dir = os.path.join(save_dir, 'Test')

        nifti_fileName = volume_name + '_' + slice_start.zfill(3) + '-' + slice_end.zfill(3) + '.nii.gz'

        nifti_address = os.path.join(nifti_dir,nifti_fileName)
        
        if not os.path.isfile(nifti_address):
            print(nifti_address + ' could not be found')
        
        niftiF = nib.load(nifti_address)


        a = np.array(niftiF.dataobj)
        y1 = int(np.round(float(bbox.split(',')[0])))
        y2 = int(np.round(float(bbox.split(',')[2])))
        x1 = int(np.round(float(bbox.split(',')[1])))
        x2 = int(np.round(float(bbox.split(',')[3])))


        # mask_file_value = 1400
        # mask = np.zeros(shape = a.shape, dtype = a.dtype)
        # mask[x1-3:x1, y1:y2, int(slice_index)-int(slice_start)] = mask_file_value
        # mask[x2:x2+3, y1:y2, int(slice_index)-int(slice_start)] = mask_file_value
        # mask[x1:x2, y1-3:y1, int(slice_index)-int(slice_start)] = mask_file_value
        # mask[x1:x2, y2:y2+3, int(slice_index)-int(slice_start)] = mask_file_value
        # vim.visualize(a, mask)

        # concatenate to handle edge cases where annotation is on the first or last slice
        a = np.concatenate((a[:,:,0][:,:,np.newaxis], a), axis = 2)
        a = np.concatenate((a, a[:,:,0][:,:,np.newaxis]), axis = 2)
        slice_index  = int(slice_index) + 1


        numpy_arr = a[:,:,int(slice_index)-int(slice_start)-1:int(slice_index)-int(slice_start)+2].astype(np.int16)
        if numpy_arr.shape != (512,512,3):
            print(numpy_arr.shape)
            input('......')

        numpy_file_name = volume_name + '_' + str(slice_index-1).zfill(4) + '_lesion_' + str(lesion_type) + '_xMin_' + str(y1).zfill(4) + '_yMin_' + str(x1).zfill(4) + '_xMax_' + str(y2).zfill(4) + '_yMax_' + str(x2).zfill(4) + '.npy'
        save_add = os.path.join(file_save_dir, numpy_file_name)
        np.save(save_add, numpy_arr)
        counter += 1
    print('counter is:' + str(counter))

if __name__ == "__main__":
    convert2NumPy(csv_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/Documents_Information/DL_info.csv',
                nifti_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/nifti_dataset', 
                save_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/numpy_dataset/3_consecutive_slice')