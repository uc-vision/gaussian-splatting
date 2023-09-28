
import argparse
from dataclasses import dataclass
from pathlib import Path

import argparse
from pathlib import Path
from typing import List
import cv2

import numpy as np
import torch
from splatting.camera.fov import FOVCamera
from tqdm import tqdm
from splatting.gaussians.renderer import render_gaussians

from splatting.gaussians.workspace import load_workspace

from ..label_transfer.instance_mask import extract_instances, InstanceMask
from ..label_transfer.detector import find_detectors, model_setup, vis_outputs


def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("model_path", type=Path)
  parser.add_argument("--detector", type=str, required=True)
  parser.add_argument("--model_name", type=str)

  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument("--show", action="store_true")
  

  args = parser.parse_args()

  detectors = find_detectors()

  if args.detector not in detectors:
    options = ", ".join(list(detectors.keys()))
    raise ValueError(f"Detector {args.detector} not found, options: {options}")    
  
  detector, metadata = model_setup(detectors[args.detector])

  workspace = load_workspace(args.model_path)

  if args.model_name is None:
      args.model_name = workspace.latest_iteration()
    
  gaussians = workspace.load_model(args.model_name).to(args.device)
  bg_color = torch.Tensor([0, 0, 0])

  for camera in workspace.cameras:
    
    resized = camera # camera.resize_shortest(detector.size_range)
    print(camera.image_name, camera.image_size, resized.image_size, detector.size_range)
    
    labels = torch.nn.Parameter(torch.zeros(gaussians.batch_shape, dtype=torch.float32, device=args.device))
    opt = torch.optim.Adam([labels], lr=0.1)

    with torch.no_grad():
      rgb = render_gaussians(resized, gaussians, bg_color).image
      bgr = rgb.flip(0)

      detections = detector(bgr * 255)
      
      if args.show:
        bgr = (bgr.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        vis = vis_outputs(detections, bgr, metadata)

        vis = cv2.resize(vis, (int(1024 * camera.aspect), 1024))
        bgr = cv2.resize(bgr, (int(1024 * camera.aspect), 1024))


        cv2.imshow("detections", vis)
        cv2.waitKey(0)

        cv2.imshow("detections", bgr)
        cv2.waitKey(0)





if __name__ == "__main__":
  main()  

