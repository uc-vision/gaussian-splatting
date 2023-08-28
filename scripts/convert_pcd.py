import argparse
from tensor_model import loading

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description="Test loading/saving of gaussians")
  parser.add_argument("input", type=str, help="Input file")

  parser.add_argument("--write_pcd", type=str, help="Write PCD file")

  args = parser.parse_args()

  gaussians = loading.read_gaussians_compat(args.input)

  if args.write_pcd:
    loading.write_gaussians(args.write_pcd, gaussians)