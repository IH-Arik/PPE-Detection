"""
Setup script for Hazard Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hazard-detection-system",
    version="1.0.0",
    author="IH Arik",
    author_email="contact@example.com",
    description="AI-powered construction hazard and PPE detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IH-Arik/Hazard-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hazard-detection=src.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.py", "utils/*.py"],
    },
    keywords="computer vision, object detection, yolo, construction safety, ppe, hazard detection",
    project_urls={
        "Bug Reports": "https://github.com/IH-Arik/Hazard-Detection/issues",
        "Source": "https://github.com/IH-Arik/Hazard-Detection",
        "Documentation": "https://github.com/IH-Arik/Hazard-Detection/blob/main/README.md",
    },
)
