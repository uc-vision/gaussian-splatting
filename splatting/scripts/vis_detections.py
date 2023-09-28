import argparse
from pathlib import Path
import cv2
import torch

from ..label_transfer.detector import find_detectors, model_setup, vis_outputs

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("image_files", nargs="+")
  parser.add_argument("--detector", type=str, required=True)

  args = parser.parse_args()

  detectors = find_detectors()

  if args.detector not in detectors:
    options = ", ".join(list(detectors.keys()))
    raise ValueError(f"Detector {args.detector} not found, options: {options}")    
  
  detector, metadata = model_setup(detectors[args.detector])


  for image_file in args.image_files:
    input_image = cv2.imread(image_file)
    
    if input_image is None:
      continue

    h, w, _ = input_image.shape
    outputs = detector(torch.from_numpy(input_image))
    vis = vis_outputs(outputs, input_image, metadata)

    vis = cv2.resize(vis, (1024, int(1024 * h / w)))
    input_image = cv2.resize(input_image, (1024, int(1024 * h / w)))


    cv2.imshow("output", vis)
    cv2.waitKey(0)

    cv2.imshow("output", input_image)
    cv2.waitKey(0)

    # torch.cuda.empty_cache()



if __name__ == "__main__":
  main()