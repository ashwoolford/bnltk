from setuptools import setup, find_packages

setup(
    name="bnltk",
    version="0.7.8",
    author="Asraf Patoary",
    author_email="asrafhossain197@gmail.com",
    description="BNLTK(Bangla Natural Language Processing Toolkit)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ashwoolford/bnltk",
    packages=find_packages(),
    install_requires=[
        "black==24.10.0",
        "keras==3.6.0",
        "numpy==2.0.2",
        "requests==2.32.3",
        "scikit-learn==1.5.2",
        "tensorflow==2.18.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True
)
