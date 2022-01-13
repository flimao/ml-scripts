from setuptools import setup

setup(
    name = 'mltoolkit',
    version = '0.2.0',    
    description = 'A useful toolkit for working with machine learning models',
    url = 'https://github.com/flimao/mltoolkit',
    author = 'LmnICE',
    author_email = 'mltoolkit@dev.lmnice.me',
    license = 'GNU GPLv3',
    packages = ['mltoolkit'],
    install_requires = [
        'numpy', 'pandas', 'typing', 'sklearn', 'matplotlib', 'seaborn', 'scipy', 'pmdarima'
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Machine Learning',
        'License :: OSI Approved :: GNU GPLv3',         
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
