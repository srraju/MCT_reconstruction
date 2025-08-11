from setuptools import setup, find_packages

setup(
    name='mct_segmentation',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scikit-image', 'pillow', 'matplotlib', 'imageio', 'opencv-python', 'pandas', 'tifffile'
    ],
    author='srraju',
    description='A Python package for MCT image segmentation and ROI extraction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/srraju',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
