from setuptools import setup, find_packages

setup(
    name='poquad-t5-utils',
    version='0.1.0',
    author='Your Name',
    author_email='m.tkacz6@student.uw.edu.pl',
    description='Package containing usefull scripts for Poleval 2024 QA Challenge using PLT5 Model from Huggingface',
    long_description=open('Readme.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LazyDart/poleval-2024-qa',
    packages=find_packages(where='scripts'),
    package_dir={'': 'scripts'},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().splitlines(),  # Reads dependencies from requirements.txt
)
