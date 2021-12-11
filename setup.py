import os
import subprocess

import setuptools
from setuptools.command.build_ext import build_ext

class CMakeExtension(setuptools.Extension):

    def __init__(self, name, sourcedir=''):
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildExt(build_ext):

    def build_extension(self, ext) -> None:
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmakedir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not cmakedir.endswith(os.path.sep):
            cmakedir += os.path.sep

        subprocess.check_call(['cmake', ext.sourcedir, '-DBUILD_PYTHON=ON', 
                            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + cmakedir],
                            cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'], cwd=self.build_temp)


setuptools.setup(
    name = 'yolov5tensorrt',
    version = '0.1',
    author = 'Noah van der Meer',
    description = 'Real-time object detection with YOLOv5 and TensorRT',
    long_description = 'file: README.md',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/noahmr/yolov5-tensorrt',
    keywords = ['yolov5', 'tensorrt', 'object detection'],
    license = "MIT",
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Linux'
    ],
    install_requires = ['numpy'],
    ext_modules = [CMakeExtension('yolov5tensorrt')],
    cmdclass = {'build_ext': CMakeBuildExt},
    scripts = [
        'examples/builder/build_engine.py',
        'examples/batch/process_batch.py',
        'examples/image/process_image.py'
    ],
    python_requires = '>=3.6'
)