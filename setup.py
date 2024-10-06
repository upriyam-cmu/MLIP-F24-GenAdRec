from distutils.core import setup

setup(
    name="gen-ad-rec",
    version="0.0.0",
    install_requires=[
        'einops',
        'numpy',
        'pillow',
        'packaging',
        'torch',
        'torchvision',
        'tqdm',
        'scikit-learn',
        'pandas'
    ],
    author="",
    author_email="",
    url="none",
    packages=["genadrec"],
)