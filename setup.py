from pathlib import Path
from setuptools import find_packages, setup

scripts = [f'{script.stem} = splatting.scripts.{script.stem}:main'
  for script in Path('splatting/scripts').glob('*.py') if script.stem != '__init__']


setup(
    name='splatting',
    version='0.1',
    packages=find_packages(),
    install_requires = [
        'camera-geometry-python',
        'open3d',
        'tqdm'
    ],

    entry_points={
      'console_scripts': scripts
    },

    include_package_data=True,
    package_data={'': ['*.ui', '*.yaml']},


)
