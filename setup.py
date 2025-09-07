from setuptools import setup, find_packages
import codecs

def readme():
    with codecs.open("README.md", "r", "utf-8") as f:
        return f.read()

setup(
    name="napari-vascilia",
    version="1.4.0",
    author="Yasmin Kassim",
    author_email="ymkgz8@mail.missouri.edu",
    description="A plugin for deep learning-based 3D analysis of cochlear hair cell stereocilia bundles.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ucsdmanorlab/Napari-VASCilia",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "napari_vascilia": ["napari.yaml", "assets/VASCilia_logo1.png"],
    },
    install_requires=[
        # exact pins to avoid source builds on Windows
        "numpy==1.26.4",
        "scikit-learn==1.3.2",
        # your other deps
        "opencv-python",
        "matplotlib",
        "imagecodecs",
        "tifffile",
        "napari[all]",
        "readlif",
        "czitools==0.4.1",
        "npe2",
        "colormap==1.1.0",
        "segmentation-models-pytorch==0.3.3",
        "pretrainedmodels==0.7.4"
    ],
    entry_points={
        "napari.manifest": [
            "napari-vascilia = napari_vascilia:napari.yaml",
        ],
    },
)
