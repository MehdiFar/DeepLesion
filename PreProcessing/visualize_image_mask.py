import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import copy
import os

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        # self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin = 0, vmax = 1)
        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin = -1000, vmax = 400)
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.ax.axes.get_yaxis().set_visible(False)
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def visualize(image, mask):
    patch = image+mask
    if image.ndim == 3:
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, patch)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
    else:
        plt.imshow(patch,cmap='gray')
        plt.show()


def visualize_dataset(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    rows = samples // cols
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, bbox, _ = dataset[i]
        image = image[:,:,1]
        # print(bbox)

        mask_file_value = 1400
        mask = np.zeros(shape = image.shape, dtype = image.dtype)
        y1, x1, y2, x2 = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]
        mask[x1-3:x1, y1:y2] = mask_file_value
        mask[x2:x2+3, y1:y2] = mask_file_value
        mask[x1:x2, y1-3:y1] = mask_file_value
        mask[x1:x2, y2:y2+3] = mask_file_value
        image = image + mask

        ax.ravel()[i].imshow(image, cmap='gray', vmin = -1000, vmax = 400)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def visual_eval(img, gt_bbox, pred_bbox):
    img = (img - (np.min(img)))/(np.max(img) - np.min(img))

    img[:,:,0] = img[:,:,1]
    img[:,:,2] = img[:,:,1]

    y1, x1, y2, x2 = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    img[x1-3:x1, y1:y2, 0] = 0
    img[x2:x2+3, y1:y2, 0] = 0
    img[x1:x2, y1-3:y1, 0] = 0
    img[x1:x2, y2:y2+3, 0] = 0    
    img[x1-3:x1, y1:y2, 1] = 1
    img[x2:x2+3, y1:y2, 1] = 1
    img[x1:x2, y1-3:y1, 1] = 1
    img[x1:x2, y2:y2+3, 1] = 1
    img[x1-3:x1, y1:y2, 2] = 0
    img[x2:x2+3, y1:y2, 2] = 0
    img[x1:x2, y1-3:y1, 2] = 0
    img[x1:x2, y2:y2+3, 2] = 0

    for box in pred_bbox:
        y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
        img[x1-3:x1, y1:y2, 0] = 1
        img[x2:x2+3, y1:y2, 0] = 1
        img[x1:x2, y1-3:y1, 0] = 1
        img[x1:x2, y2:y2+3, 0] = 1
        img[x1-3:x1, y1:y2, 1] = 0
        img[x2:x2+3, y1:y2, 1] = 0
        img[x1:x2, y1-3:y1, 1] = 0
        img[x1:x2, y2:y2+3, 1] = 0
        img[x1-3:x1, y1:y2, 2] = 0
        img[x2:x2+3, y1:y2, 2] = 0
        img[x1:x2, y1-3:y1, 2] = 0
        img[x1:x2, y2:y2+3, 2] = 0


    plt.imshow(img)
    plt.show()

def visualize_detections(img_dir, csv_file, score_thresh):
    df = pd.read_csv(csv_file)
    files = os.listdir(img_dir)
    for idx, row in df.iterrows():
        volume_name = row['uid']
        y1, x1, y2, x2 = row['xmin'], row['y_min'], row['y_max'], row['x_max']
        score = row['score']
        slice_no = volume_name.split("_")[-1][-4]
        found_files = [item for item in files if item.startswith(volume_name[:12])]
        found_files = [item for item in found_files if item.endswith(str(slice_no)+'.npy')]
        # print(len(found_files))

# img_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/numpy_dataset/3_consecutive_slice/Tune/'
# csv_file = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/Results/Dummy/2023-09-26-11:39:42.501758/detection.csv'
# visualize_detections(img_dir, csv_file, 0)

