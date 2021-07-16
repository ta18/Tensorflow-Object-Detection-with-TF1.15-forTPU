# Tensorflow-Object-Detection-with-TF1.15-forTPU
TensorFlow training scripts to perform transfer-learning on a quantization-aware object detection model and then convert it for compatibility with the Edge TPU. Specifically, this tutorial shows you how to retrain a MobileNet V1 SSD model with your own dataset, using TensorFlow r1.15.

Env : 
conda create -n tf1 python=3.6  
conda activate tf1   
#téléchargement du git   

! pip uninstall tensorflow -y  
! pip install tensorflow-gpu==1.15  
# pip install tensorflow==1.15 # for a CPU   
