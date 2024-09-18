from setuptools import setup, find_packages

setup(
    name='ic',  
    version='0.1.0',
    author='Pedro Calligaris Delbem',
    author_email='pedrodelbemo@gmail.com',
    description='Define a classe "current_circuit" as suas dependências',
    url='https://github.com/delbempedro/ic',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['math'], 
)