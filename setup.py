from setuptools import setup
from os.path import join
import sys
import versioneer

install_requires = ['menpo>=0.7,<0.8',
                    'menpo3d>=0.4,<0.5',
                    'Flask>=0.10.1',
                    'Flask-RESTful>=0.2.12',
                    'CherryPy>=3.8.0',
                    'joblib>=0.8.4',
                    'PyYaml>=3.11']

if sys.version_info.major == 2:
    install_requires.extend(['pathlib>=1.0'])

setup(name='landmarkerio',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Menpo-based server for www.landmarker.io',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      url='https://github.com/menpo/landmarkerio-server/',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
      ],
      packages=['landmarkerio'],
      install_requires=install_requires,
      scripts=[join('landmarkerio', 'lmio'),
               join('landmarkerio', 'lmioserve'),
               join('landmarkerio', 'lmiocache'),
               join('landmarkerio', 'lmioconvert')]
      )

