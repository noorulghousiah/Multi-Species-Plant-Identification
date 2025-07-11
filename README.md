
# 🌿 Multi-Family Plant Identification in Vegetation Quadrat Images using EfficientNet (PlantCLEF2024 Subset) [PlantCLEF2024](https://www.kaggle.com/competitions/plantclef-2025)

This project performs fine-grained classification of **5 families of plants : Asteraceae, Brassicaceae, Fabaceae, Poaceae, Rosaceae**.

---
# Multi-Family-Plant-Identification
- kaggle link: https://www.kaggle.com/code/ghousiah/multi-family-plant-classification?scriptVersionId=246060987
- kaggle dataset used: https://www.kaggle.com/datasets/ghousiah/plantclef/data
- [![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/code/ghousiah/multi-family-plant-classification?scriptVersionId=246060987)

---
## Faster code
- This code used saved dataset obtained from the previous code. So the there is no need for long hour for preprocessing dataset again. For example if you want to rerun the training with higher epoch number.
- Kaggle link: https://www.kaggle.com/code/ghousiah/saved-dataset-multi-family-plant-classification?scriptVersionId=246081704
- kaggle dataset used: https://www.kaggle.com/datasets/ghousiah/datasetforfivefamiliesplant/data

---
## 📁 Dataset

**Source**: `PlantCLEF2024singleplanttrainingdata.csv` size: 750M from [Download Link](https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/)

**Preprocessing**:
1) Dataset filtered to only includes labeled images from Family: {'Asteraceae': 0, 'Brassicaceae': 1, 'Fabaceae': 2, 'Poaceae': 3, 'Rosaceae': 4}.
2) The dataset images is downloaded (from url links) and cached to the notebook environments.
3) The dataset is split for training and validation in 8:2 ratio.
4) Each image is transformed into tensor, and then resize to 400 x 400.
5) After that, **quadrat images** are created by combining 4 plant images in a grid form for each quadrat image.
6) The quadrat images is resized to 96 x 96, transformed into tensor, and normalised.
7) Training and validation dataset consist of preprocessed quadrat images.

---

## 🧠 Pretrained Model
- **Base Model**: [`efficientnet_lite0`]
- **Framework**: PyTorch
- **Model Input Size**: 96 x 96
- **Model Output Class**: 5 Families
- **Modifications**:
  - Replaced classification head with `nn.Linear(in_features, 5)`
  - Frozen all other layers to speed up training on CPU

```python
model = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=NUM_CLASSES)
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True
```

---  
## **Time Taken**
1) Download and Cache Dataset Images = 2 hour
Downloading 10000 images to 'plant_cache'...
100%|██████████| 10000/10000 [2:08:28<00:00,  1.30it/s]
   
2) Training Time for 25 epochs=

Epoch 24/25 | Time: 72.27s | Train Loss: 1.3514 | Val Loss: 1.5813 | Val F1: 0.5634
Epoch 25/25 | Time: 71.76s | Train Loss: 1.3316 | Val Loss: 1.5822 | Val F1: 0.5373
✅ Final model checkpoint saved as 'final_checkpoint.pth'
⏱️ Total training time: 30.05 minutes


---  
## **Evaluation**
📊 Evaluation Metrics:
✅ F1 Score   : 0.5856
✅ Precision  : 0.6072
✅ Recall     : 0.6203
