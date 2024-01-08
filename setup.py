from setuptools import setup, find_packages

setup(
    name="tinymagic",
    version="0.0.1",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.6",
        "pandas>=1.3.5",
        "matplotlib>=3.5.3",
        "Pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "torch>=1.13.1",
        "torchvision>=0.14.1",
    ],
)
