import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import pandas as pd
import utils
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import datetime
import csv
from torchsummary import summary
import transforms as T

def read_csv_gt(csv_dir):
	df = pd.read_csv(csv_dir)
	gt_list = []
	for idx, row in df.iterrows():
		volume_name = "_".join(row['File_name'].split("_")[:-1])
		volume_name = volume_name + '_' + str(row['Key_slice_index']).zfill(4)
		# volume_name = row['File_name'][:-4]
		slice_index = row['Key_slice_index']
		bbox = row['Bounding_boxes']
		lesion_type = row['Coarse_lesion_type']
		Possibly_noisy = row['Possibly_noisy']
		
		Image_size = row['Image_size']
		if Possibly_noisy == 1: 
			continue
		if Image_size.split(',')[0] != '512':
			continue
		
		y1 = float(bbox.split(',')[0])
		y2 = float(bbox.split(',')[2])
		x1 = float(bbox.split(',')[1])
		x2 = float(bbox.split(',')[3])
		gt_list.append([volume_name, slice_index, y1, x1, y2, x2, lesion_type])
	return gt_list

class DeepLesion(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        image_filename = img_path.split('/')[-1][:-4]
        img = np.load(img_path)
        img = np.clip(img, -1024, 3071)
        img = ((img+1024)/(3071+1024))*255.0
        img = Image.fromarray(img.astype('uint8'),'RGB')
        metaData = image_filename.split('_')
        lesionType = int(metaData[5])
        boxes = []
        boxes.append([float(metaData[7]), float(metaData[9]), float(metaData[11]), float(metaData[13])])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # print(type(img))
        # input('#####')
        # draw = ImageDraw.Draw(img)
        # draw.rectangle((target["boxes"][0][0], target["boxes"][0][1], target["boxes"][0][2], target["boxes"][0][3]), outline='red')
        # img.show()
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

class DeepLesion_test(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root)))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        image_filename = img_path.split('/')[-1][:-4]
        img = np.load(img_path)
        img = np.clip(img, -1024, 3071)
        img = ((img+1024)/(3071+1024))*255.0

        img = Image.fromarray(img.astype('uint8'),'RGB')

        metaData = image_filename.split('_')
        lesionType = int(metaData[5])
        boxes = []
        boxes.append([float(metaData[7]), float(metaData[9]), float(metaData[11]), float(metaData[13])])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)


        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)
    

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")


# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)




def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def evaluate_custom2(model, data_loader, log_dir, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    stream = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        with open(os.path.join(log_dir,'detection.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['uid', 'x_min', 'y_min', 'x_max', 'y_max', 'scores'])


            for i, (images, targets, uids) in enumerate(stream, start=1):
                images = list(img.to(device) for img in images)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                outputs = model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

                for idx in range(len(images)):
                    image_names = []
                    x_min, y_min, x_max, y_max = [], [], [], []
                    probability = []

                    pred = outputs[idx]
                    img = images[idx]
                    img = (255.0 * (img - img.min()) / (img.max() - img.min())).to(torch.uint8)
                    img = img[:3, ...]
                    bbox = pred['boxes'].cpu().numpy()
                    score = pred['scores'].cpu().numpy()
                    if bbox.shape[0] == 0:
                        continue
                    uid = uids[idx].split("_")
                    uid = uid[0] + "_" + uid[1] + "_" + uid[2] + "_" + uid[3]
                    for b in range(bbox.shape[0]):
                        image_names.append(uid)
                        x_min.append(bbox[b][0])
                        y_min.append(bbox[b][1])
                        x_max.append(bbox[b][2])
                        y_max.append(bbox[b][3])
                        probability.append(score[b])
                    rows = zip(image_names, x_min, y_min, x_max, y_max, probability)
                    for row in rows:
                        writer.writerow(row)


                    # ################ next lines for visualization ###################
                    # pred_boxes = pred["boxes"]
                    # scores = pred["scores"]
                    # indices = (scores>0.2).nonzero()
                    # indices = torch.squeeze(indices)
                    # pred_boxes_select = torch.index_select(pred_boxes,0, indices)
                    # output_image = draw_bounding_boxes(img, pred_boxes_select, colors="red")
                    # print(uids[idx])
                    # plt.figure(figsize=(12, 12))
                    # plt.imshow(output_image.permute(1, 2, 0).cpu())
                    # plt.show()

def set_hyperParam():
    params = {}
    params["batch_size"] = 4
    params["num_workers"] = 4
    params["device"] = "cuda:0"
    # params["device"] = "cpu"
    params["epochs"] = 10
    params["lr"] = 0.00005
    params["momentum"] = 0.9
    params["weight_decay"] = 0.0005
    params["gamma"] = 0.1
    params["data_dir"] = "/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/numpy_dataset/3_consecutive_slice"
    return params


if __name__ == "__main__":
    hyper_params = set_hyperParam()
    log_dir = os.path.join('/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/Models', str(datetime.datetime.now()).replace(" ", "_"))
    log_dir = log_dir.replace(":", "-")
    load_data_dir = hyper_params["data_dir"]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    device = torch.device(hyper_params["device"]) if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    dataset = DeepLesion(os.path.join(load_data_dir,'Train'), get_transform(train=True))
    dataset_test = DeepLesion(os.path.join(load_data_dir, 'Tune'), get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=hyper_params["batch_size"], shuffle=True, num_workers=hyper_params["num_workers"],
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=hyper_params["batch_size"], shuffle=False, num_workers=hyper_params["num_workers"],
        collate_fn=utils.collate_fn)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=hyper_params["lr"],
                                momentum=hyper_params["momentum"], weight_decay=hyper_params["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=hyper_params["gamma"])

    num_epochs = hyper_params["epochs"]

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
    
    torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
    df = pd.DataFrame.from_dict(hyper_params, orient = 'index')
    df.to_csv(os.path.join(log_dir, "HypeParameters.csv"))
    

    test_final = DeepLesion_test(os.path.join(load_data_dir, 'Tune'), get_transform(train=False))
    data_loader_final = torch.utils.data.DataLoader(test_final, batch_size=4, shuffle=False, num_workers=32,
    collate_fn=utils.collate_fn)
    evaluate_custom2(model, data_loader_final, log_dir= log_dir, device=device)

    print("Training and Evaluation are finished and the results are stored in " + log_dir)
