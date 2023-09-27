import argparse
from pathlib import Path
import cv2

from .detector import model_setup, vis_outputs


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("image_files", nargs="+")
  parser.add_argument("--config", type=Path, required=True)

  
  args = parser.parse_args()
  predictor, metadata = model_setup(args.config)

  for image_file in args.image_files:
    input_image = cv2.imread(image_file)
    
    if input_image is None:
      continue

    h, w, _ = input_image.shape
    outputs = predictor(input_image)
    vis = vis_outputs(outputs, input_image, metadata)

    vis = cv2.resize(vis, (1024, int(1024 * h / w)))

    cv2.imshow("output", vis)
    cv2.waitKey(0)

    # torch.cuda.empty_cache()



if __name__ == "__main__":
  main()