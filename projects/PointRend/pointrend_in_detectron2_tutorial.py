# -*- coding: utf-8 -*-

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from pdb import set_trace
from tqdm import tqdm
import imageio

# import some common libraries
import numpy as np
import cv2
import torch
import os
from glob import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend

"""Now, we load a PointRend model and show its prediction."""
cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/home/yxiu/Code/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "/home/yxiu/Code/detectron2/models/model_final_ba17b9.pkl"
predictor = DefaultPredictor(cfg)

files = sorted(glob("/home/yxiu/Code/DC-PIFu/data/cape/03375/*/images/*.png"))

for infile in tqdm(files):
  
  im = cv2.imread(infile)
  outfile = infile.replace("images", "images_mask")
  os.makedirs(os.path.dirname(outfile), exist_ok=True)

  outputs = predictor(im)

  # Show and compare two predictions: 
  v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
  mask = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  if mask[:250].sum() ==0.0  and mask[-150:].sum() ==0.0:
    rgba = np.concatenate((im[:,:,::-1], 255*mask[...,None]),axis=2)[250:-150]
    imageio.imwrite(outfile, rgba.astype(np.uint8))
  # plt.imshow(point_rend_result)
  # plt.show()
  
  # set_trace()