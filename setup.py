from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(file_path: str) -> list[str]:
    """Load requirements from a file, recursively expanding ``-r`` includes."""
    base_dir = Path(__file__).parent
    requirements: list[str] = []

    def _read(path: Path) -> None:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                include_path = (path.parent / line.split(maxsplit=1)[1]).resolve()
                _read(include_path)
                continue
            requirements.append(line)

    _read((base_dir / file_path).resolve())
    return requirements


# Group every optional requirements file into a pip extras_require entry.
# After this:
#   pip install "burnit_bg[experiments] @ git+https://github.com/kirilyotov/BurnIT-BG.git"
# pulls requirements_package.txt + requirements_experiments.txt in one go.
EXTRAS = {
    "experiments":         "requirements_experiments.txt",
    "data_transformation": "requirements_data_transformation.txt",
    "data_scraping":       "requirements_data_scrapping.txt",
    "dev":                 "requirements_local_dev.txt",
    "test":                "requirements_test.txt",
}
extras_require = {name: read_requirements(path) for name, path in EXTRAS.items()}
# 'all' is the union of every extra — convenient for a full local install.
extras_require["all"] = sorted({pkg for reqs in extras_require.values() for pkg in reqs})


setup(
    name="burnit_bg",
    version="0.1.0",
    packages=find_packages(include=[
        "data_platform",
        "data_platform.*",
        "data_prep",
        "data_prep.*",
        "experiments",
        "experiments.shared",
        "experiments.shared.*",
        "utils",
        "utils.*",
    ]),
    install_requires=read_requirements("requirements_package.txt"),
    extras_require=extras_require,
    author="Kiril Yotov",
    author_email="kirilyotov@outlook.com",
    description="Pre-training LLMs on Bulgarian mental health data.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kirilyotov/BurnIT-BG",
    python_requires=">=3.12",
)
