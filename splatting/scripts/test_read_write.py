from splatting.gaussians.loading import read_gaussians, write_gaussians
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('output')
  args = parser.parse_args()

  gaussians = read_gaussians(args.input)
  write_gaussians(args.output, gaussians)
