from setuptools import setup, find_packages
#from distutils.core import Extension, setup

with open("README.md", 'r') as f:
    long_description = f.read()
	
setup(
    name="[nameproject]",
    version="1.0",
    packages=find_packages(),
	author="[author]",
    description="[desc]",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="[url]",
    setup_requires=[
        '[to complete]',
    ],
    install_requires=[
        '[to complete]',
    ],
    python_requires=">=2.7"

)
