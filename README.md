# Semantic Segmentation 
### Using UNet Model and Jupyter Notebook
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.9.7-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://www.anaconda.org/"><img src="https://img.shields.io/badge/conda-v4.10.3-blue.svg?logo=conda&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.10.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>

Image             |  Mask
:-------------------------:|:-------------------------:
![1e6f48393e17_03](https://user-images.githubusercontent.com/93069949/144007961-93770b5f-541d-4c4e-bd36-0c5cac4f2d41.jpg) | ![1e6f48393e17_03_mask](https://user-images.githubusercontent.com/93069949/144008010-61a6c9cd-eb48-426b-9351-8971c05bd0db.gif)

## Installation
1. Create conda environment
```
conda create --name env-name gitpython
```

2. Clone Github
```
from git import Repo
Repo.clone_from("https://github.com/ihamdi/Semantic-Segmentation.git","/your/directory/")
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; or [download](https://github.com/ihamdi/Semantic-Segmentation/archive/refs/heads/main.zip) and extract a copy of the files.

3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to your machine. For example:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

4. Install dependencies from [`requirements.txt`](https://github.com/ihamdi/Semantic-Segmentation/blob/main/requirements.txt) file:
```
pip install -r requirements.txt
```

5. Download Data:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Run [`python scripts/download_data.py`](https://github.com/ihamdi/Semantic-Segmentation/blob/main/scripts/download_data.py) to download the data using the Kaggle API and extract it automatically. If you haven't used Kaggle API before, please take a look at the instructions at the bottom on how to get your API key.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Otherwise, download the files from the official [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data) page and extract "train_hq.zip" to [`imgs`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data/imgs) and "train_masks.zip" to [`masks`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data/masks) folders in the [`data`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data) directory.

## Folder Structure
1. [`data`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data) directory contains [`imgs`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data/imgs) and [`masks`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data/masks) folders.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; i. [`data/imgs`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data/imgs) folder is where the images are expected to be.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ii. [`data/masks`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/data/masks) folder is where the images are expected to be.

2. [`scripts`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/scripts) directory contains [`download_data.py`](https://github.com/ihamdi/Semantic-Segmentation/blob/main/scripts/download_data.py) used to download the dataset directly from Kaggle.
3. [`unet`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/unet) directory contains UNet model.
4. [`utils`](https://github.com/ihamdi/Semantic-Segmentation/tree/main/utils) directory contains data-loading and dice-score files.


## Dataset
Data is obtained from Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) competition. Images and Masks archieves are provided in both normal and high quality. This code utilizes the train_hp.zip as well as train_masks.zip. 

There are 318 cars in the train_hq.zip archieve. Each car has exactly 16 images, each one taken at different angles. In addition, each car has a unique id and images are named according to id_01.jpg, id_02.jpg ... id_16.jpg.

## How to use
Run the following command
```
python train.py
```
The program by default will train with 5 epochs with a batch size = 1, learning rate = 0.00001, num workers = 0, scale = 0.5, mixed precision enabled, and uses 10% of the dataset for validation. You can pass the following arguments to change the default values:
1. Epochs: --epochs
2. Batch Size: --batch-size
3. Learning Rate: --learning-rate
4. Subset Size: --sample-size
5. Number of Workers: --num-workers
6. Image Scale: --scale
7. Percentage used as Validation: --validation
8. Mixed Precision: --amp

For example:
```
python train.py sample-size 500 num-workers 15 --amp
```

## Results
Dice score is printed at the end of every validation round. However, the program uses [Weights and Biases](https://wandb.ai/home) to log training loss and accuracy as well as the dice score. This makes it quite easy to visualize the results and check the status of runs without being at the training machine.

<p align="center">
  <img width="750" src="https://user-images.githubusercontent.com/93069949/144018168-bf8f72ba-040d-4f52-bf59-0426710f1755.png">
</p>

## Changes made to [Original Code](https://github.com/milesial/Pytorch-UNet)
1. Fixed data download problem from Kaggle. The code no longer gives an "unauthorized" error.
2. Introduced sample_size to enable reduction of dataset if needed (mainly used for testing).
3. Added num_workers as a variable so it doesn't need to be changed manually inside train.py.
4. Added sample_size and num_workers to logging.
5. Added sample_size and num_workers to arguments so they can be set easily when calling python train.py.
6. Changed RMSProp to Adam optimizer for better results.
7. Fixed progress bar for training. Original code show a fixed number of total iterations even if batch size is changed. The update was also choppy and only happening every 2 iterations.
8. Fixed validation loop. Now it runs at the end of each epoch.
9. Removed commented out lines and unused import statements.

---

#### Background:

This was created to learn about semantic segmentation and UNet, therefore only the training data is utilized.

---

#### Contact:

For any questions or feedback, please feel free to post comments or contact me at ibraheem.hamdi@mbzuai.ac.ae

---

### Referernces:

[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) was used as base for this code.

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, Thomas Brox

---

#### *Getting Key for Kaggle's API

![image](https://user-images.githubusercontent.com/93069949/144188576-d457568e-7cd2-42f2-ba08-9c41143d674d.png)

![image](https://user-images.githubusercontent.com/93069949/144188635-705e1e29-92ae-4aba-be66-0e1d2e1c29ca.png)

![image](https://user-images.githubusercontent.com/93069949/144188696-f535f9c8-3ed8-4e1b-8f0d-179d7e5be2a2.png)
