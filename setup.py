from setuptools import setup, find_packages

setup(
    name="ollama_python",
    version="0.1.0",
    author="Richard Ogunyale",
    author_email="kogunyale01@gmail.com",
    description="Python Wrapper around Ollama API Endpoints",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kennyrich/ollama-python",
    packages=find_packages(),
    install_requires=[
        "httpx >=0.26.0",
        "pydantic >=2.5.3",
        "requests>=2.31.0",
        "responses >=0.24.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
