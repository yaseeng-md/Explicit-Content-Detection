# Explicit Content Detection using Vision Transformers

##  Overview
This repository focuses on detecting explicit content in images using **Vision Transformers (ViTs)** and **Swin Transformers**. The project explores multiple architectures, including convolution-based patch extraction and autoencoder-based feature extraction, to improve detection accuracy.

## Models Implemented
1. **Feed Forward Patch Extraction** - Standard ViT approach with feed-forward networks.
2. **Convolution-Based Patch Extraction** - Using CNNs to extract patches before feeding them into transformers.
3. **Swin Transformer Model** - A hierarchical vision transformer achieving **98.5% accuracy**.
4. **Autoencoder Feature Extraction (Upcoming)** - Enhancing feature representation with autoencoders.

## How to Use This Repository
Clone the repository using:
```bash
  git clone https://github.com/yaseeng-md/Explicit-Content-Detection
  cd Explicit-Content-Detection
```
Install the necessary dependencies from the requirements.txt file.
```bash
pip install -r requirements.txt
```
Run the provided Jupyter notebooks to preprocess the data, train models, and make predictions.

## Prerequisits
Before you continue with the implemntation, Take the help of **Dataset Collection.py** and **Remove Corrupt Files.py** in Helpers.


**Remove Corrupt Files.py** Helps you to remove corrupted files from the exitsing dataset.


**Dataset Collection.py** You can decide the amount of dataset that you want to train and experiment on.

## Trained Models
[Download the trained Models from here !](https://drive.google.com/drive/folders/1pXR-1hBxhV8rD6jXxoPAVCsCMDK78iyB?usp=sharing)

## Dataset
Provided on request.

##  Results
| Model                      | Traning Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|----------------------------|------------------|---------------|---------------------|-----------------|
| ViT (Feed Forward)         | 78.95%           | 0.4638        | 81.89%              | 0.4351          |
| ViT (CNN Patch Extractor)  | 88.05%           | 0.430         | 81.86%              | 0.4893          |
| Swin Transformer           | 98.35%           | 0.0458        | 86.89%              | 0.3443          |


##  Future Work
- Implement **Feed Forward Autoencoder-based Feature Extraction**.
- Optimize **Convution Autoencoder-based Feature Extraction**.

