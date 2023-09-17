from dataclasses import dataclass
from typing import List
from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt, QEvent

import numpy as np
import pyrender

from viewer.camera import FlyCamera
from viewer.scene import Scene

from tensor_model.fov_camera import FOVCamera
from tensor_model.gaussians import Gaussians
    
@dataclass 
class Settings:
  update_rate : int = 60
  move_speed : float = 1.0
  rotate_speed : float = 1.0
  


class SceneWidget(QtWidgets.QWidget):
  

  def __init__(self, gaussians:Gaussians, cameras:List[FOVCamera]):
    super(SceneWidget, self).__init__()

    SceneWidget.instance = self

    self.scene = Scene()
    self.state = FlyCamera()

    self.timer = QtCore.QTimer(self)
    self.timer.timeout.connect(self.update)
    self.timer.start(1000 / Settings.update_rate)

    self.settings = Settings()

    self.renderer = pyrender.OffscreenRenderer(
      self.size().width(), self.size().height())
    
    self.gaussians = gaussians
    self.cameras = cameras

    self.scene.set_camera(self.cameras[0])

    points = pyrender.Mesh.from_points(self.gaussians.positions, gaussians.colors)
    self.scene.add(points)

    # image_size = (self.size().width(), self.size().height())
    # print(self.cameras[0].fov, self.scene.get_camera(image_size).fov)

    # self.scene.set_camera(self.scene.get_camera(image_size))
    # print(self.scene.get_camera(image_size))

    self.setFocusPolicy(Qt.StrongFocus)
    self.setMouseTracking(True)

  def sizeHint(self):
    return QtCore.QSize(1024, 768)


  def event(self, event: QEvent):
    if not self.state._event(event):
      return super(SceneWidget, self).event(event)

    return False

  def update(self):
    self.state._update(1 / Settings.update_rate)
    self.repaint()


  def render(self):
    return self.renderer.render(self.scene.scene)

  def paintEvent(self, event: QtGui.QPaintEvent):
    with QtGui.QPainter(self) as painter:
      image, depth = self.render()
      image = np.ascontiguousarray(image)


      painter.drawImage(0, 0, QtGui.QImage(image, image.shape[1], image.shape[0], 
                                           QtGui.QImage.Format_RGB888))

  def resizeEvent(self, event: QtGui.QResizeEvent):
    w, h = event.size().width(), event.size().height()
    # Round width up to multiple of 4
    w = (w + 3) & ~0x03

    self.renderer = pyrender.OffscreenRenderer(w, h)
    return True



    