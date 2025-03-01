# Explicit Content Detection using Vision Transformers

## 📌 Overview
This repository focuses on detecting explicit content in images using **Vision Transformers (ViTs)** and **Swin Transformers**. The project explores multiple architectures, including convolution-based patch extraction and autoencoder-based feature extraction, to improve detection accuracy.

## 🚀 Models Implemented
1. **Feed Forward Patch Extraction** - Standard ViT approach with feed-forward networks.
2. **Convolution-Based Patch Extraction** - Using CNNs to extract patches before feeding them into transformers.
3. **Swin Transformer Model** - A hierarchical vision transformer achieving **98.5% accuracy**.
4. **Autoencoder Feature Extraction (Upcoming)** - Enhancing feature representation with autoencoders.

## 📂 Project Structure
```
📦 Explicit-Content-Detection
 ┣ 📂 datasets          # Dataset preprocessing scripts
 ┣ 📂 models            # Model implementations (ViT, Swin Transformer, etc.)
 ┣ 📂 utils             # Helper functions for training & evaluation
 ┣ 📜 train.py          # Model training script
 ┣ 📜 test.py           # Evaluation script
 ┣ 📜 requirements.txt  # Dependencies list
 ┗ 📜 README.md         # Project documentation
```

## 🔧 Installation
### Prerequisites
Ensure you have Python installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Model
Run the following command to train the model:
```bash
python train.py --model vit --epochs 50
```
Replace `vit` with `swin` or other model names as needed.

## 📊 Performance Evaluation
To evaluate the trained model on a test dataset:
```bash
python test.py --model swin
```

## 📜 Results
| Model                      | Traning Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|----------------------------|------------------|---------------|---------------------|-----------------|
| ViT (Feed Forward)         | 78.95%           | 0.4638        | 81.89%              | 0.4351          |
| ViT (CNN Patch Extractor)  | 88.05%           | 0.430         | 81.86%              | 0.4893          |
| Swin Transformer           | 98.35%           | 0.0458        | 86.89%              | 0.3443          |

## 📌 Future Work
- Implement **Autoencoder-based Feature Extraction**.
- Optimize **dataset preprocessing techniques**.
- Deploy the model as a **Flask or FastAPI application**.

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve the project.

## 📜 License
This project is licensed under the MIT License.
