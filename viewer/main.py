import sys
import argparse

from PySide6 import QtGui, Qt3DRender, QtCore, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget
from tensor_model.loading import read_gaussians
from tensor_model.workspace import load_workspace

from viewer.scene_widget import SceneWidget



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
    if parsed_args.model_name is None:
      parsed_args.model_name = workspace.latest_iteration()
    
    print("Loading", parsed_args.model_name)
    gaussians = read_gaussians(workspace.cloud_files[parsed_args.model_name])

    qt_args = sys.argv[:1] + unparsed_args
    app = QApplication(qt_args)
    

    window = QtWidgets.QMainWindow()

    scene_widget = SceneWidget(gaussians, workspace.cameras)
    window.setCentralWidget(scene_widget)

    window.show()
    sys.exit(app.exec())

  
if __name__ == '__main__':
  main()