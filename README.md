# Visual Altimeter

This should introduce you to the basics of the project. After reading this go to [Wiki pages](https://github.com/DSSVisualAltimeter/DSS-visual-altimeter/wiki) for more documentation.  

# Description

Currently, drones measure the altitude using GPS, barometers, or internal navigation systems. The goal of this repository is to document and investigate the ability to use a drone's camera footage as an altitude measurement.  


## Usage

This repository is intended to collaborate and communicate work.  
There are two main parts to this project that contributors will work on.  
1. Data Collection - involves using python to generate data from Airsim (Airsim installation required)  
2. Model Development - involves using python for creating the model. Eventually, this will require access to a GPU for traning

## Installation

Different parts of the project may have different requirements. Find which one you want to contribute to and follow the guidelines  

### Data Collection

1. Make sure you have Python Downloaded and an environment setup either through pipenv, venv, conda
2. [Get started Downloading Airsim](https://github.com/DSSVisualAltimeter/DSS-visual-altimeter/wiki/Installation-Airsim-on-Windows)
3. Consult Project Manager to know where data is stored

### Model Development

1. Make sure you have Python Downloaded and an environment setup either through pipenv, venv, conda  
2. Find out where the data is  
3. Ensure Pytorch or Tensorflow/Keras is downloaded  

## File Structure

Here are the basics of how the repository is set up

- `personal-folders`: contributors may use this to hold all python scripts that they are developing but do not feel are ready to be used by anyone.  
- `scripts`: python files that are used to complete the project
- `images`: contains grayscale images generated from Airsim by Lawrence Chilton
- `data`: currently holds nothing
- `archive_project_Fall_2019`: contains all files uploaded from September - December 2019


## Helpful Links
1. [bulk data storage](https://drive.google.com/drive/folders/1Wm8eyhZ8ujNSy5ReifbYFk4cvICGkRuH?usp=sharing)
Data for the project will be stored here
 
2. [Microsoft Airsim Official Documentation](https://github.com/Microsoft/AirSim/)

3. [Anaconda](https://www.anaconda.com/distribution/) 
A python distribution for data science.

4. To learn more about how to use Github go to the [Git Documentation](https://git-scm.com/doc)

