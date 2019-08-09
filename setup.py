from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='perscode',
      version='0.0.1',
      description='Representation of persistence diagrams using persistence codebooks',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Carlos Ronchi and Lun Zhang',
      author_email='carloshvronchi@gmail.com and chamberlian1990@gmail.com',
      url='https://github.com/chronchi/perscode',
      license='MIT',
      packages=['perscode'],
      include_package_data=True,
      install_requires=[
        'scikit-learn',
        'numpy',
      ],
      extras_require={ # use `pip install -e ".[testing]"``
        'testing': [
          'pytest'
        ],
        'docs': [ # `pip install -e ".[docs]"``
          'sktda_docs_config'
        ]
      },
      python_requires='>=2.7,!=3.1,!=3.2,!=3.3',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
      ],
      keywords="""persistent homology, persistence codebooks, persistence diagrams, topological
                  data analysis, algebraic topology"""
     )
