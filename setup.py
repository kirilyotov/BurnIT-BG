from setuptools import setup, find_packages


setup(
    name="burnit_bg",
    version="0.1.0",
    packages=find_packages(include=[
        "data_platform",
        "data_platform.*",
        "utils",
        "utils.*",
    ]),
    install_requires=open("requirements_package.txt", encoding="utf-8").read().splitlines(),
    author="Kiril Yotov",
    author_email="kirilyotov@outlook.com",
    description="Pre-training LLMs on Bulgarian mental health data.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kirilyotov/BurnIT-BG",
    python_requires=">=3.14",
)