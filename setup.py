from setuptools import setup, find_packages

setup(
    name="transitlens-ml-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "astropy>=5.3",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "matplotlib>=3.7",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "httpx>=0.25",
        ]
    },
    author="Team TransitLens",
    description="AI-enabled detection and classification of exoplanet transit signals from astronomical light curves",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "transitlens=core.cli:main",
        ]
    },
)
