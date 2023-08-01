from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = ["mkdocs==1.3.0", "mkdocstrings==0.18.1"]

style_packages = ["black==22.3.0", "flake8==6.0.0", "Flake8-pyproject==1.2.3", "isort==5.12.0"]

dev_packages = ["pre-commit==3.3.3", "typer==0.9.0"]

setup(
    name="tagifai",
    version=0.1,
    description="Classify machine learning projects.",
    author="Mitchell Read",
    author_email="mitchell.g.read@gmail.com",
    python_requires=">=3.10",
    install_requires=[required_packages],
    extras_require={"dev": docs_packages + style_packages + dev_packages, "docs": docs_packages},
)
