from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

def parse_requirements(path="requirements.txt"):
    reqs = []
    p = HERE / path
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs

setup(
    name="worm_analytic_suite",
    version="1.0.0",
    description="Python framework for analysis of worm silhouettes",
    long_description=(HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Christian Pritz",
    packages=find_packages(where="code"),   # 👈 search inside the 'code' folder
    package_dir={"": "code"},               # 👈 tell setuptools that packages live in 'code'
    include_package_data=True,
    install_requires=parse_requirements(),
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov", "flake8", "black", "ruff"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
