import sys
import argparse

from PySide6 import QtGui, Qt3DRender, QtCore, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget

from viewer.scene_widget import SceneWidget



import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class Window(QtWidgets.QMainWindow):
  def __init__(self):
    super().__init__()

    self.render = SceneWidget()
    self.setCentralWidget(self.render)



def process_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')  # positional argument

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args

def main():
    parsed_args, unparsed_args = process_cl_args()

    qt_args = sys.argv[:1] + unparsed_args
    app = QApplication(qt_args)
    

    view = Window()
    view.show()
    sys.exit(app.exec())

  
if __name__ == '__main__':
  main()