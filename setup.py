from setuptools import setup

setup(
    name="llmner",
    version="0.1.0",
    description="# LLMNER: Named Entity Recognition without training data",
    url="https://github.com/plncmm/llmner",
    author="PLN@CMM",
    author_email="fabian.villena@uchile.cl",
    license="BSD 2-clause",
    packages=["llmner"],
    install_requires=["openai==0.28.1", "langchain==0.0.321", "nltk==3.8.1"],
    extras_require={"dev": ["ipykernel==6.25.2"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
