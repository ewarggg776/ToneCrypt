from setuptools import setup

# Read README.md for long_description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tonecrypt',
    version='0.1',
    author='Ewarggg776',
    author_email='gilroyandrew6@gmail.com',
    description='A tool for encrypting files into audio tones',
    long_description=long_description,
    long_description_content_type='text/markdown',
    scripts=['ToneCrypt.py'],
    data_files=[
        ('/usr/share/tonecrypt', ['README.md', 'LICENSE']),
        ('/usr/share/tonecrypt/text', []),
        ('/usr/share/tonecrypt/audio', []),
    ],
    install_requires=[
        'numpy==1.24.4',
        'scipy==1.10.1',
        'soundfile==0.12.1',
        'librosa==0.10.1',
        'reedsolo==1.7.0',
        'colorama==0.4.6',
        'pydub==0.25.1',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ],
)