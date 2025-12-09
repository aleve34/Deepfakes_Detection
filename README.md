
# Deepfake Detection with Xception

## Overview
This project focuses on **deepfake detection** using the ** Xception** network.  
We fine-tuned the pre-trained model on the **FaceForensics++ C32 dataset**, performing binary classification (Real vs Fake). The notebook contains code, experimental results, and visualizations to understand model performance.

---

## Framework & Dependencies
- **Framework:** PyTorch  
- **Pre-trained model:** Legacy Xception (via `timm`)  
- **Python packages:** 
  - torch
  - torchvision
  - timm
  - PIL
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

> Recommended to use a GPU (Google Colab, Kaggle, or local CUDA-enabled GPU).

---

## Dataset
- **Source:** [FaceForensics++ C23 Dataset](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23)  
- **Total folders in dataset:** 6 (Original, Deepfakes, FaceSwap, NeuralTextures, DeepfakesCompression, and others)  
- **Folders used for this project:** `Original` and `Deepfakes` only  
- **Details:**  
  - Images are full frames from videos
  - Balanced dataset: ~5000 real and 5000 fake frames  
- **Preprocessing:** minimal augmentation (random horizontal flip only)  
- **Note:** The dataset is too large to include in this repository. You can download it from the link above.

---

## Model Architecture
- **Xception**:  
  - Depthwise separable convolutions  
  - Entry, middle, and exit flows  
  - Global average pooling  
  - Fully connected classifier modified for 2 classes  
- **Fine-tuning:** last 5 blocks + classifier head

---

## Training Details
- **Optimizer:** Adam  
- **Loss function:** Cross-Entropy  
- **Scheduler:** StepLR (step=5, gamma=0.5)  
- **Batch size:** 16  
- **Epochs:** 20 (with early stopping)  
- **Device:** GPU recommended  

---

## Metrics & Results
- **Validation Accuracy:** 91.20%  
- **Test Accuracy:** 93.47%  
- **Classification Report:**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Real  | 0.9179    | 0.9547 | 0.9359   |
| Fake  | 0.9528    | 0.9147 | 0.9333   |

- **Observations:**  
  - Balanced precision and recall for both classes  
  - Confusion matrix shows few misclassifications  
  - Model generalizes well on test set  

---

## Usage
1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo>
```
2. Install dependencies:
```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn pillow
```

3. Place the dataset in the folder /FF++C32-Frames or update the path in the notebook.

3. Run the notebook to train or evaluate the model.
