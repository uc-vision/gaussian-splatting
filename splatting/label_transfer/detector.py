import argparse
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.modeling import build_model


from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from pathlib import Path
import cv2

from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo

import numpy as np
import torch

from .instance_mask import draw_instances, extract_instances

def vis_outputs(outputs, image, dataset_metadata):
  visualizer = Visualizer(image[:, :, ::-1],
              metadata=dataset_metadata,
              scale=1,
              instance_mode=ColorMode.SEGMENTATION)
  
  if 'panoptic_seg' in outputs:
    out = visualizer.draw_panoptic_seg(
      outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1])
  elif 'instances' in outputs:
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

  return out.get_image()[:, :, ::-1]


 
def draw_panoptic(input_image, outputs, metadata):
  labels = outputs['sem_seg'].argmax(dim=0).to("cpu").numpy()
  colors = np.array(metadata.stuff_colors, dtype=np.uint8)

  vis =  (colors[labels].astype(np.float32) + input_image.astype(np.float32)) / 512
  vis = (vis * 255).astype(np.uint8)
  vis[labels == 0] = input_image[labels == 0]

  instances = extract_instances(outputs)
  return draw_instances(vis, instances, metadata.thing_colors)


def find_detectors():
  model_dir = Path(__file__).parent / "models"
  return {file.stem:file for file in list(model_dir.glob("*.json"))}


class Predictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.size_range = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image:torch.Tensor):
        with torch.no_grad():  
            if image.shape[-1] == 3:
              image = image.permute(2, 0, 1)

            if self.input_format == "RGB":
                image = image.flip(0)

            height, width = image.shape[-2:]
            image = image.to(self.cfg.MODEL.DEVICE).to(torch.float32)

            inputs = {"image": image,
                      "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions






def model_setup(model_config:Path, device:str="cuda:0"):

  with open(model_config, 'r') as f:
    model_info = json.load(f)

  dataset_name = model_info['dataset_name']
  metadata = MetadataCatalog.get(dataset_name)

  if 'thing_classes' in model_info:
    metadata.thing_classes = model_info['thing_classes']

  if 'thing_colors' in model_info:
    metadata.thing_colors = model_info['thing_colors']
  elif hasattr(metadata, 'thing_colors'):
    metadata.thing_colors = np.random.randint(0, 255, (len(metadata.thing_classes), 3))
  
  if 'stuff_classes' in model_info:
    metadata.stuff_classes = model_info['stuff_classes']

  if 'stuff_colors' in model_info:
    metadata.stuff_colors = model_info['stuff_colors']
  elif hasattr(metadata, 'stuff_colors'):
      metadata.stuff_colors = np.random.randint(0, 255, size = (len(metadata.stuff_colors), 3))
  
  cfg = get_cfg()

  if 'model_zoo' in model_info:
    cfg.merge_from_file(model_zoo.get_config_file(model_info['model_zoo']))
  elif 'config_file' in model_info:
    filename = model_config.parent / model_info['config_file']
    cfg.merge_from_file(filename)
  else:
    raise ValueError("No config file specified")

  cfg.MODEL.DEVICE = device

  cfg.MODEL.WEIGHTS = str(model_config.parent / model_info['weight_file'])
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
  
  if 'image_size' in model_info:
    image_size = model_info['image_size']

    cfg.INPUT.MAX_SIZE_TEST =  image_size
    cfg.INPUT.MIN_SIZE_TEST = image_size

  if 'score_thresh_test' in model_info: 
    conf_threshold = model_info.get('score_thresh_test', 0.1)

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf_threshold

  return Predictor(cfg), metadata

