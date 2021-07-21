[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

# Tensorflow-Object-Detection-with-TF1.15-forTPU

  ![objectDetection](imageDetection.png)     

TensorFlow training scripts to perform transfer-learning on a quantization-aware object detection model and then convert it for compatibility with the Edge TPU. Specifically, this tutorial shows you how to retrain a MobileNet V1 SSD model with your own dataset, using TensorFlow 1.15.

## HardWare specifications 



## 1. Clone the repository :
In a shell download the *tod_tf1* repository on your computer (I recommande you to clone the repository in your home directory $HOME)   
`git clone https://github.com/ta18/tod_tf1`

And create the PYTHON PATH :   
`nano ~/.bashrc`  
Add this line in the end :   
`export TOD_ROOT="$HOME/tod_tf1"     
export PYTHONPATH=$TOD_ROOT/models:$TOD_ROOT/models/research:$TOD_ROOT/models/research/slim:$PYTHONPATH  
alias tf1="conda activate tf1"`  
You have to change the TOD_ROOT with your own path (it's the place that you have clone the repository) and source your the basrc :    
`source ~/.bashrc`

## 2. Create the virtual environnement :   
You have to create a virtual environnement with python 3.6 :
`conda create -n tf1 python=3.6`  
When the environnement it's create, enter in :    
`conda activate tf1`  or just `tf1` thanks to the alias in .bashrc file
Then, install Tensorflow 1.15 :     
`pip uninstall tensorflow -y`    
`pip install tensorflow-gpu==1.15` 
If you work on a CPU :   
`pip install tensorflow==1.15`    

Your environnement it's ready ! 

## 3. Go on your virtual environnement on a new shell :   
Open a new shell and go here : 
`cd ~/tod_tf1` enter the path of your tod_tf1 folder    

Write this command to enter to your environnement : 
`tf1`   

Open the notebook *Retrain SSD mobilnet for object detection.ipynb* with jupyter notebook :     
`jupyter notebook`   
A web page open, and you have to open the *Retrain SSD mobilnet for object detection.ipynb* notebook.  
For now follow this notebook.   

Training notebook : https://github.com/ta18/tod_tf1/blob/main/Retrain%20SSD%20mobilnet%20for%20object%20detection.ipynb    
Use the training network notebook : https://github.com/ta18/tictactoe2021/blob/main/notebooks/test_formDetection_tf1.ipynb   
