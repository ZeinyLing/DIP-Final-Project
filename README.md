# Watermark Removal: A Weakly Supervised Approach Using Multi-View Visual Models and Inpainting
<img src="pictures/pic1.png" width="900">
<img src="pictures/pic2.png" width="900">
## Requirements
```
pip install -r requirements.txt
```
## Datasets From kaggle
[Kaggle watermark dataset](https://doi.org/10.34740/KAGGLE/DSV/5811178)
Download the train part, and split it into train, val , test by random
## Step0.Image Preprocessing to Multi-View
```
python step1_train.py
```
## Step1.Training model
According to your file name, modify 
```
# data_dir = './dataset/' <--- According to your file name , modify it in code

#run code
python step1_train.py
```
## Step2.Voting
```
python step2_voting.py
```
## Step3.CAM to Mask
```
python step1_train.py
```

## Step4.Inpaint on Colab
[Main Inpaint code (Stable-Diffusion-Inpaint)](https://colab.research.google.com/drive/1TiPyjSF8TU-NQiozjsXROZE5zkQGJrcr?usp=sharing)
```
python step1_train.py
```



## Step5.Inpaint on Colab
