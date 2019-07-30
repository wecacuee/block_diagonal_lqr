from setuptools import setup, find_packages
setup(name='block-diagonal-lqr',
      packages=find_packages(),
      install_requires=open('requirements.txt').readlines(),
      long_description=open('README.md', encoding='utf-8').read(),
      long_description_content_type="text/markdown")
