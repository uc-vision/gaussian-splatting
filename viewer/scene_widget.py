from dataclasses import dataclass
from typing import List, Optional
from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt, QEvent
import PySide6.QtGui
import cv2

import math

import numpy as np
import pyrender
import torch
from tensor_model.workspace import Workspace

from viewer.camera import FlyCamera
from viewer.scene import Scene

from pathlib import Path

from tensor_model.fov_camera import FOVCamera
from tensor_model.gaussians import Gaussians
from tensor_model.renderer import render_gaussians
    
@dataclass 
class Settings:
  update_rate : int = 60
  move_speed : float = 1.0
  rotate_speed : float = 2.0

  drag_speed : float = 1.0
  point_size : float = 2.0

  snapshot_scale: float = 16.0

  render_multiple : int = 16
  device : str = 'cuda:0'
  





class SceneWidget(QtWidgets.QWidget):
  

  def __init__(self, workspace:Workspace, model_name:Optional[str] = None, settings:Settings = Settings()):
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
    
    self.workspace = workspace

    if model_name is None:
      model_name = workspace.latest_iteration()
    
    # self.initial = workspace.load_initial_points()
    # points = pyrender.Mesh.from_points(
    #   self.initial.point['positions'].numpy(), self.initial.point['colors'].numpy())
    
    # self.scene.add(points, pose=np.eye(4))

    self.gaussians = workspace.load_model(model_name)

    points = pyrender.Mesh.from_points(self.gaussians.positions, self.gaussians.colors)
    self.scene.add(points, pose=np.eye(4))

    self.gaussians = self.gaussians.to(self.settings.device)

    camera = self.workspace.cameras[0]
    self.scene.set_fov_camera(camera)
    print('Showing view from camera', camera.image_name)

    self.setFocusPolicy(Qt.StrongFocus)
    self.setMouseTracking(True)

  @property
  def image_size(self):
    return self.size().width(), self.size().height()

  def sizeHint(self):
    return QtCore.QSize(1024, 768)


  def event(self, event: QEvent):
    if not self.state.trigger_event(event):
      return super(SceneWidget, self).event(event)

    return False
  
  def keyPressEvent(self, event: QtGui.QKeyEvent) -> bool:
    if event.key() == Qt.Key_Print:
      self.save_snapshot()
      return True

    
    return super().keyPressEvent(event)

  def update(self):
    self.state._update(1 / Settings.update_rate)
    self.repaint()

  def render(self, camera):
    with torch.no_grad():

      rendering = render_gaussians(camera, self.gaussians, torch.Tensor([0, 0, 0]))
      
      image = (rendering.image.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8)
      image = image.cpu().numpy()
      return np.ascontiguousarray(image)
    # return self.renderer.render(self.scene.scene)

  def paintEvent(self, event: QtGui.QPaintEvent):
    with QtGui.QPainter(self) as painter:

      m = self.settings.render_multiple 
      def next_mult(x):
        return int(math.ceil(x / m) * m)
        

      w, h = self.image_size
      round_size = (next_mult(w), next_mult(h))
      image = self.render(self.scene.get_fov_camera(round_size))
      


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



    