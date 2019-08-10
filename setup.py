from pathlib import Path
from setuptools import setup, find_packages

scriptdir=Path(__file__).parent
def scriptrel(relfile, scriptdir=scriptdir):
    return str(scriptdir.joinpath(relfile))


setup(name='block-diagonal-lqr',
      packages=find_packages(),
      install_requires=open(scriptrel('requirements.txt')).readlines(),
      tests_requires=['pytest'],
      long_description=open(scriptrel('README.md'), encoding='utf-8').read(),
      long_description_content_type="text/markdown")
