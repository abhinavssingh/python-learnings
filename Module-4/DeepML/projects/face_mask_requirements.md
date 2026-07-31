# Capstone Session 10 Requirements

# Project Title

Face Mask Detection Using Transfer Learning

---

# Business Objective

Build a Deep Learning Transfer Learning solution capable of detecting whether a person is:

- Wearing a face mask correctly
- Not wearing a face mask
- Wearing a face mask incorrectly

The objective is to classify images into one of three categories and identify the best-performing transfer learning architecture.

---

# Dataset Description

## Image Dataset

The dataset consists of face images grouped into three classes:

| Class               | Description                |
| ------------------- | -------------------------- |
| with_mask           | Face mask worn correctly   |
| without_mask        | No face mask               |
| mask_worn_incorrect | Face mask worn incorrectly |

Dataset metadata includes:

| Variable        | Description          |
| --------------- | -------------------- |
| Image File Name | Image filename       |
| Class           | Image category label |

All images have dimensions:

```text
128 x 128 x 3
```

(RGB images) 【1-71b2a5】

---

# Dataset Structure

```text
dataset/
│
├── train/
│   ├── with_mask/
│   ├── without_mask/
│   └── mask_worn_incorrect/
│
└── test/
    ├── with_mask/
    ├── without_mask/
    └── mask_worn_incorrect/
```

---

# Task A

## EfficientNetB0 Transfer Learning Model

### Data Loading

Load training and testing images from:

```text
train/
test/
```

Use:

```python
ImageDataGenerator
```

### Training Dataset

Use:

```python
validation_split=0.2
```

Create:

```python
Training Dataset
Validation Dataset
```

### Test Dataset

Use:

```python
ImageDataGenerator
```

without augmentation.

---

# Model Architecture

Use:

```python
EfficientNetB0
```

as the base model.

### Architecture

````text
EfficientNetB0
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.2)
        ↓
Dense(3, activation="softmax")
```

---

# Compilation

Use:

```python
optimizer = Adam
````

Loss:

```python
categorical_crossentropy
```

Metrics:

```python
accuracy
```

---

# Training

Epochs:

```python
25
```

Callbacks:

## EarlyStopping

Monitor:

```python
val_loss
```

## ReduceLROnPlateau

Monitor:

```python
val_loss
```

---

# Visualization

Plot:

## Accuracy

```text
Training Accuracy
Validation Accuracy
```

vs

```text
Epochs
```

## Loss

```text
Training Loss
Validation Loss
```

vs

```text
Epochs
```

---

# Task B

## ResNet50 Transfer Learning Model

### Dataset Preparation

Repeat the same preprocessing steps used in Task A.

Use:

```python
ImageDataGenerator
```

Training:

```python
validation_split=0.2
```

Testing:

```python
ImageDataGenerator
```

---

# Model Architecture

Use:

```python
ResNet50
```

as the base model.

### Architecture

```text
ResNet50
      ↓
GlobalAveragePooling2D
      ↓
Dropout(0.5)
      ↓
Dense(3, activation="softmax")
```

---

# Compilation

Optimizer:

```python
Adam
```

Loss:

```python
categorical_crossentropy
```

Metric:

```python
accuracy
```

---

# Training

Epochs:

```python
25
```

Callbacks:

```python
EarlyStopping
ReduceLROnPlateau
```

Monitor:

```python
val_loss
```

---

# Evaluation

Using the best trained model:

Predict all images in:

```text
test/
```

Generate:

- Accuracy
- Classification Metrics
- Predictions【1-71b2a5】

---

# Test Image Visualization

Display:

```text
10 Test Images
```

For each image display:

- Image
- True Label
- Predicted Label

---

# Task C

## Model Comparison

Compare:

- EfficientNetB0
- ResNet50

Evaluation Criteria:

- Validation Accuracy
- Test Accuracy
- Loss
- Generalization Performance

Determine:

```text
Best Performing Model
```

---

# Final Prediction

Using the best model:

Predict on the test dataset.

Display:

```text
10 Sample Images
```

along with:

```text
True Label
Predicted Label
```

---

# Deliverables

## Data Preparation

- ImageDataGenerator
- Train Dataset
- Validation Dataset
- Test Dataset

## EfficientNetB0 Model

- Transfer Learning
- Training
- Evaluation

## ResNet50 Model

- Transfer Learning
- Training
- Evaluation

## Visualizations

- Accuracy Curve
- Loss Curve

## Prediction Results

- 10 Test Images
- Predicted Labels
- True Labels

## Model Comparison

- EfficientNetB0 vs ResNet50

## Best Model Selection

- Final Recommendation

---

# Expected Output

A complete Transfer Learning solution capable of:

- Detecting face masks
- Handling three-class classification
- Comparing EfficientNetB0 and ResNet50
- Selecting the best-performing model
- Visualizing predictions and performance metrics
