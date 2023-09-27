
import argparse
from dataclasses import dataclass
from pathlib import Path

import argparse
from pathlib import Path
from typing import List
import cv2

import numpy as np
from tqdm import tqdm

from splatting.gaussians.workspace import load_workspace

from ..label_transfer.instance_mask import extract_instances, InstanceMask
from ..label_transfer.detector import find_detectors, model_setup


@dataclass
class ImageInstances:
  instances : List[InstanceMask]
  image : np.ndarray


def load_detect_image(detector, image_path:Path):
  image = cv2.imread(str(image_path))
  outputs = detector(image)

  return extract_instances(outputs)



  
  

def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("model_path", type=Path)
  parser.add_argument("--detector", type=str, required=True)
  


  args = parser.parse_args()

  detectors = find_detectors()

  if args.detector not in detectors:
    options = ", ".join(list(detectors.keys()))
    raise ValueError(f"Detector {args.detector} not found, options: {options}")    
  
  detector, metadata = model_setup(detectors[args.detector])
 

  workspace = load_workspace(args.model_path)

  if model_name is None:
      model_name = workspace.latest_iteration()
    
  gaussians = workspace.load_model(model_name)

  print("Detecting images...")
  image_masks = {k:load_detect_image(detector, camera.image_name) 
                 for k, camera in tqdm(workspace.cameras.items())}




  # image_files = 


if __name__ == "__main__":
  main()  

