
from typing import Optional
from PySide6 import QtGui
from PySide6.QtCore import QEvent

from beartype import beartype

class Interaction():
  def __init__(self):
    super(Interaction, self).__init__()
    self.child = None 
    self.active = False

  def transition(self, interaction:Optional['Interaction']):
    self.pop()
    if interaction is not None:
      self.push(interaction)

  def push(self, interaction:'Interaction'):
    self.child = interaction
    if self.active:
      self.child._activate()


  def pop(self):    
    if self.child is not None:
      child = self.child
      self.child = None
      child._deactivate()

  def transition(self, interaction:Optional['Interaction']):
    self.pop()
    if interaction is not None:
      self.push(interaction)

  def _activate(self):
    self.on_activate()

    if self.child is not None:
      self.child._activate()

    self.active = True


  def _deactivate(self):
    if self.child is not None:
      child = self.child
      self.child = None
      child._deactivate()

    self.on_deactivate()


  def _event(self, event: QEvent) -> bool:
    if self.child is not None:
      if self.child._event(event):
        return True
      
    return self.event(event) 
  
  def _update(self, dt:float) -> bool:
    if self.child is not None:
      if self.child._update(dt):
        return True
    
    return self.update(dt)

  @beartype
  def event(self, event: QEvent) -> bool:
    event_callbacks = {
      QEvent.KeyPress: self.keyPressEvent,
      QEvent.KeyRelease: self.keyReleaseEvent,
      QEvent.MouseButtonPress: self.mousePressEvent,
      QEvent.MouseButtonRelease: self.mouseReleaseEvent,
      QEvent.MouseMove: self.mouseMoveEvent,
      QEvent.Wheel: self.wheelEvent,
      QEvent.FocusIn: self.focusInEvent,
      QEvent.FocusOut: self.focusOutEvent,
    }

    if event.type() in event_callbacks:
      return event_callbacks[event.type()](event) or False
    
    return False
    


  def keyPressEvent(self, event: QtGui.QKeyEvent):
    return False

  def keyReleaseEvent(self, event: QtGui.QKeyEvent):
    return False

  def mousePressEvent(self, event: QtGui.QMouseEvent):
    return False

  def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
    return False

  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    return False

  def wheelEvent(self, event: QtGui.QWheelEvent):
    return False

  def focusInEvent(self, event: QtGui.QFocusEvent):
    return False
  
  def focusOutEvent(self, event: QtGui.QFocusEvent):
    return False

  @beartype
  def update(self, dt) -> bool:
    return False

  def on_activate(self):
    pass

  def on_deactivate(self):
    pass

  @property
  def scene_widget(self):
    from .scene_widget import SceneWidget
    return SceneWidget.instance
  
  @property
  def scene(self):
    return self.scene_widget.scene
  
  
  @property
  def settings(self):
    return self.scene_widget.settings
