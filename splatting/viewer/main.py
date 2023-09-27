import sys
import argparse

from PySide6 import QtGui, Qt3DRender, QtCore, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget

from splatting.gaussians.workspace import load_workspace

from splatting.gaussians.gaussians import Gaussians
from .scene_widget import SceneWidget, Settings


import signal





def process_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')  # positional argument
    parser.add_argument('--model_name', default=None)  # positional argument
    parser.add_argument('--device', default='cuda:0')
    

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args

def sigint_handler(*args):
    QApplication.quit()


def main():
    signal.signal(signal.SIGINT, sigint_handler)


    parsed_args, unparsed_args = process_cl_args()
    workspace = load_workspace(parsed_args.model_path)

    if parsed_args.model_name is None:
      parsed_args.model_name = workspace.latest_iteration()

    gaussians = workspace.load_model(parsed_args.model_name)
    print(f"Loaded model {parsed_args.model_name}: {gaussians}")


    qt_args = sys.argv[:1] + unparsed_args
    app = QApplication(qt_args)
    

    window = QtWidgets.QMainWindow()

    scene_widget = SceneWidget(settings=Settings(device=parsed_args.device))
    scene_widget.load_workspace(workspace, gaussians)

    window.setCentralWidget(scene_widget)

    window.show()
    sys.exit(app.exec())

  
if __name__ == '__main__':
  main()