"""
PyMTCNN - High-Performance MTCNN Face Detection for Apple Silicon
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="pymtcnn",
    version="1.0.0",
    author="SplitFace",
    description="High-performance MTCNN face detection optimized for Apple Neural Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/PyMTCNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "coremltools>=7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pillow>=9.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    package_data={
        "pymtcnn": [
            "models/*.mlpackage/**/*",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    license="CC BY-NC 4.0",
    keywords=[
        "face detection",
        "mtcnn",
        "coreml",
        "apple neural engine",
        "computer vision",
        "deep learning",
    ],
)
