from dataclasses import dataclass
from typing import Optional
from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt, QEvent
import PySide6.QtGui

from beartype import beartype

import numpy as np
from viewer.interaction import Interaction


class FlyCamera(Interaction):
  def __init__(self):
    super(FlyCamera, self).__init__()
    self.down = set()
    self.drag_mouse_pos = None

    self.directions = { 
      Qt.Key_Q : np.array([0.,  -1.,  0.]),
      Qt.Key_E : np.array([0.,  1.,  0.]),
      
      Qt.Key_W : np.array([0.,  0.,  -1.]),
      Qt.Key_S : np.array([0.,  0.,  1.]),

      Qt.Key_A : np.array([-1., 0.,  0.]),
      Qt.Key_D : np.array([1.,  0.,  0.])
    }


  def keyPressEvent(self, event: QtGui.QKeyEvent):
    if event.key() in self.directions and not event.isAutoRepeat():
      self.down.add(event.key())
      return True
      
      
  def keyReleaseEvent(self, event: QtGui.QKeyEvent):
    if event.key() in self.directions  and not event.isAutoRepeat():
      self.down.discard(event.key())
      return True
    
  def focusOutEvent(self, event: QtGui.QFocusEvent):
    self.down.clear()
    return True
  

  def update(self, dt:float):
    scene = self.scene
    for key in self.down:
      scene.move_camera(self.directions[key] * dt * self.settings.move_speed)

  def mousePressEvent(self, event: QtGui.QMouseEvent):
    if event.buttons() & Qt.LeftButton:
      self.drag_mouse_pos = event.localPos()
      return True

  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    if event.buttons() & Qt.LeftButton and self.drag_mouse_pos is not None:
      delta = event.localPos() - self.drag_mouse_pos

      sz = self.scene_widget.size()

      
      self.scene.rotate_camera(delta.x() / sz.width() * self.settings.rotate_speed, 
                                delta.y() / sz.height() * self.settings.rotate_speed)
      
      self.drag_mouse_pos = event.localPos()
    