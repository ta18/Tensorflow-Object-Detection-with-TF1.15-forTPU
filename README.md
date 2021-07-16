# Tensorflow-Object-Detection-with-TF1.15-forTPU
TensorFlow training scripts to perform transfer-learning on a quantization-aware object detection model and then convert it for compatibility with the Edge TPU. Specifically, this tutorial shows you how to retrain a MobileNet V1 SSD model with your own dataset, using TensorFlow r1.15.

## Clone the repository :
In an other shell dowload the repository where you want (but you have to keep this path for after):
`git clone https://github.com/ta18/tod_tf1`

And create the PYTHON PATH :   
`nano ~/.bashrc`  
Add this line in the end :   
`export TOD_ROOT="/home/jlc/tod_tf1"  
export PYTHONPATH=$TOD_ROOT/models:$TOD_ROOT/models/research:$TOD_ROOT/models/research/slim:$PYTHONPATH  
alias tf1="conda activate tf1"`  
You have to change the TOD_ROOT with your own path (it's the place that you have clone the repository)  

## Create the virtual environnement :   
`conda create -n tf1 python=3.6`  
`conda activate tf1`  
Install Tensorflow 1.15 :   
`pip uninstall tensorflow -y`  
`pip install tensorflow-gpu==1.15`  
If you work on a CPU :   
`pip install tensorflow==1.15`     

## Go on your virtual environnement on a new shell :   
`tf1`  
Open the notebook *Retrain SSD mobilnet for object detection.ipynb* with jupyter notebook :   
`jupyter notebook`  
A web page open, and you have to open the *Retrain SSD mobilnet for object detection.ipynb* notebook.  
For now follow this notebook.   
