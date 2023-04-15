from setuptools import setup

setup(
    name='hri_tools',
    version='0.0.8',
    description='A toos for working with datasets',
    packages=['hri_tools'],
    author='Alexander Baranov',
    author_email='alexanderbaranof@gmail.com',
    url='https://github.com/Humor-Research-Institute/hri_tools',
    install_requires=[
       "pandas >= 1.3.5",
       "numpy >= 1.21.5",
       "nltk >= 3.5",
       "gdown >= 4.6.0"
   ]
)