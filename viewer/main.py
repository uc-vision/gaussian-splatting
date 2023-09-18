import sys
import argparse

from PySide6 import QtGui, Qt3DRender, QtCore, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget
from tensor_model.loading import read_gaussians
from tensor_model.workspace import load_workspace

from viewer.scene_widget import SceneWidget, Settings



import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)



def process_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')  # positional argument
    parser.add_argument('--model_name', default=None)  # positional argument
    parser.add_argument('--device', default='cuda:0')
    

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args

def main():
    parsed_args, unparsed_args = process_cl_args()

    workspace = load_workspace(parsed_args.model_path)
    

    qt_args = sys.argv[:1] + unparsed_args
    app = QApplication(qt_args)
    

    window = QtWidgets.QMainWindow()

    scene_widget = SceneWidget(workspace, 
          model_name=parsed_args.model_name, settings=Settings(device=parsed_args.device))

    window.setCentralWidget(scene_widget)

    window.show()
    sys.exit(app.exec())

  
if __name__ == '__main__':
  main()