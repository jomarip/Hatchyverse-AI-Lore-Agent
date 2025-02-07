from setuptools import setup, find_packages

setup(
    name="hatchyverse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "faiss-cpu",
        "openai",
        "python-dotenv",
        "networkx",
        "pandas"
    ],
) 