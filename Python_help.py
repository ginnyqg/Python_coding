#import csv
train = pd.read_csv("directory")

#print dim of data
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

#print head of the dataset
print(train.head())

#check current working directory(in python or jupyter notebook)
import os
os.getcwd()

#check system path
import sys
sys.path


#check virtual env version in commandline
virtualenv --version

#create a virtual env
cd my_project_folder
virtualenv my_project

#activate virtual env
source my_project/bin/activate
 
pip install requests

###############    install packages in virtual env      ###############
pip install -U numpy scipy scikit-learn


#######################################################################



deactivate

#check python package version in commandline
Python
>>> import tensorflow
>>> print(tensorflow.__version__)


#outside of virtual environment
#jupyter path problem
python3 -m pip install --upgrade pip
python3 -m pip install jupyter


#import matplotlib error
#inside virtual env at terminal
cd ~/.matplotlib
nano matplotlibrc

#type
backend: TkAgg

#Crtl + O to save, Crtl + X to exit


#import seaborn error, import in terminal, not in jupyter notebook
#cause and solution: https://github.com/jupyter/notebook/issues/2359
# check in notebook

sys.executable
# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6


sys.path
# ['',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python36.zip',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/lib-dynload',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/extensions',
#  '/Users/qinqingao/.ipython']



#reinstall seaborn in where sys.executable is
#in virtualenv
# sys.executable -m pip install seaborn

/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6 -m pip install --upgrade seaborn



