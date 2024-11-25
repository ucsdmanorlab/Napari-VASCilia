from setuptools import setup, find_packages
import codecs

def readme():
    with codecs.open("README.md", "r", "utf-8") as f:
        return f.read()

setup(
    name="napari-vascilia",
    version="1.3.0",
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
            "napari_vascilia": ["assets/VASCilia_logo1.png"],
        },
    install_requires=[
        "torch==1.12.1+cu113",
        "torchvision==0.13.1+cu113",
        "torchaudio==0.12.1+cu113",
        "segmentation-models-pytorch",
        "opencv-python",
        "matplotlib",
        "imagecodecs",
        "tifffile",
        "napari[all]",
        "scikit-learn",
        "readlif",
        "czitools==0.4.1",
        "czifile",
        "npe2"
    ],
    entry_points={
        "napari.plugin": [
            "napari_vascilia = napari_vascilia.Napari_VASCilia_v1_3_0:initialize_vascilia_ui",
        ],
    },
)
