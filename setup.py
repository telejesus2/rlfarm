from setuptools import setup, find_packages


def get_install_requires():
    install_requires = []
    with open('requirements.txt') as f:
        for req in f:
            install_requires.append(req.strip())
    return install_requires


setup(
    name='rlfarm',
    author='Jesus Bujalance',
    packages=find_packages(),
    # install_requires=get_install_requires()
)