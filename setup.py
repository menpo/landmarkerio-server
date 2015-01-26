from setuptools import setup
from os.path import join
import sys
import versioneer

project_name = 'landmarkerio'

# Versioneer allows us to automatically generate versioning from
# our git tagging system which makes releases simpler.
versioneer.VCS = 'git'
versioneer.versionfile_source = '{}/_version.py'.format(project_name)
versioneer.versionfile_build = '{}/_version.py'.format(project_name)
versioneer.tag_prefix = 'v'  # tags are like v1.2.0
versioneer.parentdir_prefix = project_name + '-'  # dirname like 'menpo-v1.2.0'

install_requires = ['menpo>=0.4.a3',
                    'numpy>=1.9.0',
                    'Flask>=0.10.1',
                    'Flask-RESTful>=0.2.11',
                    'CherryPy==3.6.0',
                    'joblib>=0.8.2']

if sys.version_info.major == 2:
    install_requires.extend(['enum>=0.4.4', 'pathlib>=1.0'])

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
