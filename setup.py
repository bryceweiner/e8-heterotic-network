"""
Setup script for E8×E8 Heterotic Network PyPi package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_readme():
    """Read README file."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    """Read requirements.txt file."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Package metadata
NAME = 'e8-heterotic-network'
VERSION = '0.1.0'
DESCRIPTION = 'E8×E8 Heterotic Structure for Geometric Deep Learning'
LONG_DESCRIPTION = read_readme()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
AUTHOR = 'E8 Heterotic Network Team'
AUTHOR_EMAIL = 'contact@e8-heterotic-network.org'
URL = 'https://github.com/e8-heterotic-network/e8-heterotic-network'
LICENSE = 'MIT'

# Classifiers for PyPI
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Framework :: PyTorch',
]

# Keywords
KEYWORDS = [
    'theoretical-physics',
    'geometric-deep-learning',
    'e8-lattice',
    'string-theory',
    'heterotic-string',
    'lie-algebra',
    'root-system',
    'network-geometry',
    'clustering-coefficient',
    'pytorch',
    'machine-learning'
]

# Required packages
INSTALL_REQUIRES = [
    'numpy>=1.20.0',
    'scipy>=1.7.0',
    'torch>=1.10.0',
    'networkx>=2.6.0',
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'black>=21.0.0',
        'isort>=5.0.0',
        'flake8>=3.9.0',
        'mypy>=0.800',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'visualization': [
        'matplotlib>=3.5.0',
        'plotly>=5.0.0',
        'seaborn>=0.11.0',
    ],
    'geometric': [
        'scikit-learn>=1.0.0',  # For PCA projections
    ],
    'all': [
        'pytest>=6.0.0',
        'matplotlib>=3.5.0',
        'plotly>=5.0.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
    ]
}

# Entry points for command line scripts
ENTRY_POINTS = {
    'console_scripts': [
        'e8-heterotic-clustering=e8_heterotic.cli.compute_clustering:main',
    ],
}

# Package data
PACKAGE_DATA = {
    'e8_heterotic': [
        'tests/*.py',
        'utils/*.py',
        'core/*.py',
    ],
}

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,

    # Package configuration
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    include_package_data=True,
    package_data=PACKAGE_DATA,

    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # Metadata
    classifiers=CLASSIFIERS,
    keywords=' '.join(KEYWORDS),

    # Entry points
    entry_points=ENTRY_POINTS,

    # Additional metadata
    python_requires='>=3.8',
    zip_safe=False,

    # Project links
    project_urls={
        'Documentation': 'https://e8-heterotic-network.readthedocs.io/',
        'Source': 'https://github.com/e8-heterotic-network/e8-heterotic-network',
        'Tracker': 'https://github.com/e8-heterotic-network/e8-heterotic-network/issues',
        'Changelog': 'https://github.com/e8-heterotic-network/e8-heterotic-network/blob/main/CHANGELOG.md',
    },
)