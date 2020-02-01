"""
File Goal:
Experiment with Resnet

@Author: Matt Jeffs
@Date: January 2020

Computer: Macbook Pro - OSX Catalina
Python Version 3.6

Tasks to be done

- set working directory (done)
- read packages (done)
- read data
    - Set up 10 folders with 10 classifications
- preprocess data
    - TBD
- implement Resnet
    - Documentation: https://keras.io/applications/#resnet
    - Pytorch Documentaion: https://pytorch.org/hub/pytorch_vision_resnet/

Resources
- Google Search: Keras Resnet youtube and view the videos for implementation

"""

# Load Packages
import os  # For navigating directories
from sys import platform as _platform  # validate platform
import sys # Ensure System version is



# Postprocessing Data
def main():
    def establish_directory():
        """
        This function allows you to set working directory
        """
        if _platform == "linux" or _platform == "linux2" or _platform == "darwin":
            # linux and Mac
            '''
            It has yet to be tested for linux but it should work as mac is a Unix application
            '''

            while True:
                try:
                    print("\nYou are currently in", os.getcwd())
                    directory_path = str(input("Enter File path of your folder (i.e. ~/fake/project_directory: "))
                    os.chdir(directory_path)
                    print("\nDirectory successfully changed to: ", directory_path)

                    break
                except AttributeError:
                    print(
                        "Directory NOT changed, check to make sure " +
                        "you have the right file path and that you " +
                        "have not mispelled anything... Try again")
                except FileNotFoundError:
                    print("No such file or directory: Could you double check the file/ directory?")
        elif _platform == "win32" or _platform == "win64":
            # Windows 10
            """
            This has yet to be tested
            """
            while True:
                try:
                    print("\nYou are currently in", os.getcwd())
                    directory_path = str(input("Enter File path of your folder (i.e. C:\...\project_directory: "))
                    os.chdir(directory_path)
                    print("\nDirectory successfully changed to: ", directory_path)

                except AttributeError:
                    print(
                        "Directory NOT changed, check to make sure " +
                        "you have the right file path and that you " +
                        "have not mispelled anything... Try again")
                except FileNotFoundError:
                    print("No such file or directory: Could you double check the file/ directory?")
        else:
            print("This is an unrecognized platform, Mac, Windows32bitt and Windows64bits are acceptable")

    def get_information():
        """
        Developing
        This function will gather the data that the directory is in and directory you want to validate the data in

        ### Make sure there are trailing slashes!! ###
        """
        data_dir = str(input("Please enter the directory the data will be in " +
                             "(i.e. drive/My Drive/Data Science Society/Visual Altimeter Video/frames/data/"))
        validation_dir = str(input("Where is your validation directory? " +
                                   "(i.e. drive/My Drive/Data Science Society/Visual Altimeter Video/frames/validation/"))

    def get_data():
        """
        This function needs to locate where the data is
        """
        pass

    """ Execute functions here"""
    establish_directory()


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")
    main()
