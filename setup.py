from pathlib import Path
from setuptools import setup, find_packages
from logging import basicConfig, getLogger, DEBUG
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

scriptdir=Path(__file__).parent
def scriptrel(relfile, scriptdir=scriptdir):
    LOG.debug(list(scriptdir.iterdir()))
    return str(scriptdir.joinpath(relfile))


setup(name='block-diagonal-lqr',
      version='0.1.0',
      packages=find_packages(),
      install_requires=open(scriptrel('requirements.txt')).readlines(),
      #install_requires=['numpy', 'matplotlib', 'scipy'],
      tests_requires=['pytest'],
      package_data={
          '': ['*.txt', '*.md'],
      },
      long_description=open(scriptrel('README.md'), encoding='utf-8').read(),
      long_description_content_type="text/markdown")
