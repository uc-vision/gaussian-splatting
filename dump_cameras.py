import argparse
import scene
from scene.gaussian_model import GaussianModel



def main():
  parser = argparse.ArgumentParser(description='Dump cameras from a scene.')
  parser.add_argument('--source_path', type=str, required=True, help='Path to the scene.')  
  parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')

  parser.add_argument('--resolution', type=float, default=0.25, help='Image resolution.')

  args = parser.parse_args()

  scene.Scene(args, gaussians=None)


if __name__ == "__main__":
  main()