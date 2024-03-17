import os
import imutils
from imutils import paths
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
from nn_drop_detector import DropZoneModel, DropZoneDataset, ToTensor
from torchvision import datasets, models
from pre_trainer import resnet18_custom

import matplotlib.pyplot as plt
import pdb

from PIL import Image

dropzone_model_weights = ""
classification_weights = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Give it image or images (stitching)

# #A. stitch images
# folder_path = ''
# imagePaths = sorted(list(paths.list_images(folder_path)))
# images = []

# for imagePath in imagePaths:
# 	image = cv2.imread(imagePath)
# 	images.append(image)
# stitcher = cv2.Stitcher_create()
# (status, stitched) = stitcher.stitch(images)
# if status == cv2.Stitcher_OK:
#     cv2.imshow("stitched img", stitched)

# #B. remember GPS locations
# #

#Run drop detector model

img_dir = ""
blank_metadata_dir = ""

test_dataset = DropZoneDataset(image_dir=img_dir, metadata_dir=blank_metadata_dir, transform=ToTensor())
img_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

drop_zone_model = DropZoneModel().to(device)
drop_zone_model.load_state_dict(dropzone_model_weights)
classifier = resnet18_custom(8, 8, 36).to(device)
classifier.load_state_dict(classification_weights)

predicted_locations = None

drop_zone_model.eval()

image = test_dataset[0]
inputs, gt_lbl = image['image'], image['drop_locations']
inputs = inputs.unsqueeze(0)
pred_location = DropZoneModel(inputs)
square_half_len = 25
pred_location = pred_location.squeeze()

image = inputs.squeeze().cpu().numpy().transpose(1,2,0)
image = np.ascontiguousarray(image)

rect_points = []

#Extract Bounding Boxes (bbx)

for loc in pred_location:
    st = (int(loc[0]) - square_half_len, int(loc[1]) - square_half_len)
    ed = (int(loc[0]) + square_half_len, int(loc[1]) + square_half_len)
    rect_points.append((st, ed))
    #cv2.rectangle(image, st, ed, (0, 0, 230), 10)

rects = []

for i in rect_points:
    rect = image[st[1]:ed[1], st[0]:ed[0], :]
    rects.append(rect)

#Run Detection model on each bbx.
    
classifier.eval()

classifications = []

for i in rects:
    classifications.append(classifier(i))




