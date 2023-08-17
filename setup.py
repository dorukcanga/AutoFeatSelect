from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name="AutoFeatSelect",
    version="0.1.2",
    packages=find_packages(),
    author="Doruk Canga",
    author_email="dorukcanga@gmail.com",
    description="Automated Feature Selection & Feaure Importance Calculation Framework",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/dorukcanga/AutoFeatSelect",
    license="MIT",
    install_requires=[
        "category_encoders",
        "xgboost",
        "lightgbm",
        "Boruta"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)