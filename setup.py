from setuptools import setup
import os.path as p

setup(name='landmarkerio-server',
      version='0.0.6',
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
      packages=['landmarkerioserver'],
      install_requires=['menpo>=0.2.5',
                        'Flask>=0.10.1',
                        'Flask-RESTful>=0.2.11'],
      scripts=[p.join('landmarkerioserver', 'landmarkerio')]
      )
