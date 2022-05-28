from setuptools import setup

setup(
    name="TransformersTSA",
    version="DEV",
    description="Transformers for Time Series Analysis",
    author="Benoit de Kersabiec",
    author_email="benoit.dekersabiec@student-cs.fr",
    install_requires=[
        "requests>=2.27.1,<2.28.0",
        "pandas>=1.4.2,<1.5.0",
        "gym>=0.23.1,<0.24.0",
        "sacred>=0.8.2,<0.9.0",
        "numpy>=1.22.3,<1.23.0",
        "torch",
        # "--pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
    ],
    python_requires=">=3.9.12,<3.10"
)
