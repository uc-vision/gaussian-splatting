from dataclasses import dataclass
from enum import Enum
import signal
import sys
from typing import List, Optional
from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt, QEvent
import PySide6.QtGui
import cv2

import math

from pathlib import Path
from scipy.spatial import distance_matrix

import numpy as np
import pyrender
import torch


from splatting.gaussians.workspace import Workspace
from splatting.gaussians.gaussians import Gaussians
from splatting.gaussians.renderer import render_gaussians
    
from .camera import FlyCamera
from .scene import Scene
from .keyboard import keyevent_to_string


@dataclass 
class Settings:
  update_rate : int = 60
  move_speed : float = 1.0
  rotate_speed : float = 2.0

  zoom_discrete : float = 1.2
  zoom_continuous : float = 0.1

  drag_speed : float = 1.0
  point_size : float = 2.0

  snapshot_scale: float = 16.0

  render_multiple : int = 16
  device : str = 'cuda:0'
  
class ViewMode(Enum):
  Initial = 0
  Gaussians = 1
  Points = 3


class SceneWidget(QtWidgets.QWidget):
  

  def __init__(self, settings:Settings = Settings()):
    super(SceneWidget, self).__init__()

    SceneWidget.instance = self

    self.scene = Scene()
    self.state = FlyCamera()

    self.timer = QtCore.QTimer(self)
    self.timer.timeout.connect(self.update)
    self.timer.start(1000 / Settings.update_rate)

    self.settings = settings

    self.renderer = pyrender.OffscreenRenderer(
      *self.image_size, point_size=self.settings.point_size)
    
    self.workspace = None
    self.gaussians = None

    self.setFocusPolicy(Qt.StrongFocus)
    self.setMouseTracking(True)


  def load_workspace(self, workspace:Workspace, gaussians:Gaussians):
    self.workspace = workspace

    points = pyrender.Mesh.from_points(gaussians.positions.cpu(), gaussians.colors.cpu())
    self.points_node = self.scene.add(points, pose=np.eye(4))

    initial = workspace.load_initial_points()
    points = pyrender.Mesh.from_points(
      initial.point['positions'].numpy(), initial.point['colors'].numpy())
    self.initial_node = self.scene.add(points, pose=np.eye(4))

    
    self.gaussians = gaussians.to(self.settings.device)
    camera_positions = np.array([c.position for c in self.workspace.cameras])
    self.settings.move_speed = np.linalg.norm(camera_positions.max(axis=0) - camera_positions.min(axis=0)) / 10.

    self.set_camera_index(0)
    self.set_viewmode(ViewMode.Gaussians)

  def set_camera_index(self, index:int):
    camera = self.workspace.cameras[index]
    print('Showing view from camera', camera.image_name)
    self.zoom = 1.0

    self.scene.set_fov_camera(camera)
    self.camera_index = index



  @property
  def image_size(self):
    return self.size().width(), self.size().height()

  def sizeHint(self):
    return QtCore.QSize(1024, 768)


  def event(self, event: QEvent):
    if not self.state.trigger_event(event):
      return super(SceneWidget, self).event(event)
    return False
  
  def set_viewmode(self, viewmode:ViewMode):
    self.view_mode = viewmode

    for node in [self.initial_node, self.points_node]:
      if self.scene.scene.has_node(node):
        self.scene.scene.remove_node(node)

    if viewmode == ViewMode.Initial:
      self.scene.add_node(self.initial_node)
    elif viewmode == ViewMode.Points:
      self.scene.add_node(self.points_node)

  
  def keyPressEvent(self, event: QtGui.QKeyEvent) -> bool:

    if event.key() == Qt.Key_Print:
      self.save_snapshot()
      return True
  
    elif event.key() == Qt.Key_BraceLeft:
      self.set_camera_index((self.camera_index - 1) % len(self.workspace.cameras))
      return True
    elif event.key() == Qt.Key_BraceRight:
      self.set_camera_index((self.camera_index + 1) % len(self.workspace.cameras))
      return True
    elif event.key() == Qt.Key_Equal: 
      camera = self.scene.get_fov_camera(self.image_size)
      self.scene.set_fov_camera(camera.zoom(self.settings.zoom_discrete))
      return True
    elif event.key() == Qt.Key_Minus:
      camera = self.scene.get_fov_camera(self.image_size)
      self.scene.set_fov_camera(camera.zoom(1 / self.settings.zoom_discrete))
      return True

    elif event.key() == Qt.Key_2:
      self.set_viewmode(ViewMode.Gaussians)
      return True
    elif event.key() == Qt.Key_1:
      self.set_viewmode(ViewMode.Initial)
      return True
    elif event.key() == Qt.Key_3:
      self.set_viewmode(ViewMode.Points)
      return True

    
    return super().keyPressEvent(event)

  def update(self):
    self.state._update(1 / Settings.update_rate)
    self.repaint()

  def render(self, camera):
    if self.view_mode == ViewMode.Gaussians:
      with torch.no_grad():
        rendering = render_gaussians(camera, self.gaussians, torch.Tensor([0, 0, 0]))
        
        image = (rendering.image.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8)
        image = image.cpu().numpy()
        return image
    else:
      image, depth = self.renderer.render(self.scene.scene)
      return image
    
  def paintEvent(self, event: QtGui.QPaintEvent):
    with QtGui.QPainter(self) as painter:

      m = self.settings.render_multiple 
      def next_mult(x):
        return int(math.ceil(x / m) * m)
        
      w, h = self.image_size
      round_size = (next_mult(w), next_mult(h))
      image = np.ascontiguousarray(
        self.render(self.scene.get_fov_camera(round_size)))
      
      painter.drawImage(0, 0, QtGui.QImage(image, image.shape[1], image.shape[0],  
                                           QtGui.QImage.Format_RGB888))
  def snapshot_file(self):
    pictures = Path.home() / "Pictures"
    filename = pictures / "snapshot.jpg"

    i = 0
    while filename.exists():
      i += 1
      filename = pictures / f"snapshot_{i}.jpg"

    return filename


  def save_snapshot(self):
    w, h = self.image_size
    scale = self.settings.snapshot_scale
    snapshot_size = (int(w * scale), int(h * scale))

    camera = self.scene.get_fov_camera(snapshot_size)
    filename = self.snapshot_file()
    print(f"Capturing snapshot {filename} at {snapshot_size[0]}x{snapshot_size[1]}")

    image = cv2.cvtColor(self.render(camera), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


  def resizeEvent(self, event: QtGui.QResizeEvent):
    w, h = event.size().width(), event.size().height()
    # Round width up to multiple of 4
    w = (w + 3) & ~0x03

    self.renderer = pyrender.OffscreenRenderer(w, h, point_size=self.settings.point_size)
    return True



def show_workspace(workspace:Workspace, gaussians:Gaussians = None):
  from splatting.viewer.main import sigint_handler
  signal.signal(signal.SIGINT, sigint_handler)

  app = QtWidgets.QApplication(["viewer"])
  widget = SceneWidget()

  if gaussians is None:
    gaussians = workspace.load_model(workspace.latest_iteration())

  print(f"Showing model from {workspace.model_path}: {gaussians}")


  widget.load_workspace(workspace, gaussians)
  widget.show()
  app.exec_()
    