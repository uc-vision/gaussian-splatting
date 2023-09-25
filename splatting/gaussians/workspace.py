from pathlib import Path
import re
from natsort import natsorted
from dataclasses import dataclass

from splatting.camera.fov import FOVCamera, load_camera_json

from .loading import read_gaussians
from .gaussians import Gaussians

import open3d as o3d 

@dataclass
class Workspace:
  model_path:Path

  cloud_files : dict[str, Path]
  cameras:dict[str, FOVCamera]

  def latest_iteration(self) -> str:
    paths = [(m.group(1), name) for name in self.cloud_files.keys()
             if (m:=re.search("iteration_(\d+)", name))]
    
    if len(paths) == 0:
      raise FileNotFoundError(f"No point clouds named iteration_(\d+) in {str(self.model_path)}")

    paths = sorted(paths, key=lambda x: int(x[0]))
    return paths[-1][1]
            
  def load_model(self, model_name:str) -> Gaussians:
    if not model_name in self.cloud_files:
      options = list(self.cloud_files.keys())
      raise LookupError(f"Model {model_name} not found in {self.model_path} options are: {options}")

    return read_gaussians(self.cloud_files[model_name])
  
  def load_initial_points(self) -> o3d.t.geometry.PointCloud:
    return o3d.t.io.read_point_cloud(str(self.model_path / "input.ply"))
  

def find_clouds(p:Path):
  clouds = {d.name : file for d in p.iterdir() 
            if d.is_dir() and (file :=d / "point_cloud.ply").exists()
  }

  if len(clouds) == 0:
    raise FileNotFoundError(f"No point clouds found in {str(p)}")

  return clouds


def load_workspace(model_path:Path | str) -> Workspace:
  model_path = Path(model_path)
  cloud_files = find_clouds(model_path / "point_cloud")

  cameras = load_camera_json(model_path / "cameras.json")
  cameras = natsorted(cameras.values(), key=lambda x: x.image_name)

  return Workspace(
    model_path = model_path,
    cloud_files = cloud_files,
    cameras = cameras
  )