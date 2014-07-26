from setuptools import setup
from os.path import join

setup(name='landmarkerio-server',
      version='0.0.7',
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
      install_requires=['menpo>=0.3.0',
                        'Flask>=0.10.1',
                        'enum>=0.4.4',
                        'Flask-RESTful>=0.2.11',
                        'CherryPy>=3.5.0',
                        'numpy>=1.8.1',
                        'pathlib>=1.0',
                        'joblib>=0.8.2'],
      scripts=[join('landmarkerio', 'lmio'),
               join('landmarkerio', 'lmioserve'),
               join('landmarkerio', 'lmiocache')]
      )
