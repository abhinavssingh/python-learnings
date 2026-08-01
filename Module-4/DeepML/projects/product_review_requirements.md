# Capstone Session 11 Requirements

# Project Title
Customer Product Review Classification Using CNN-LSTM

## Business Objective
Build a CNN-LSTM hybrid deep learning model to classify customer product reviews as Good or Bad based on review text and ratings.

Source dataset: GrammarandProductReviews.csv.

## Dataset Description

Key columns include:
- reviews.text
- reviews.rating
- reviews.title
- reviews.doRecommend
- reviews.didPurchase
- brand
- manufacturer
- categories
- reviews.username
- reviews.userCity
- reviews.userProvince

The primary features used for modeling are:
- reviews.text
- reviews.rating

## Task A

### Data Loading
- Load GrammarandProductReviews.csv.

### Target Creation
Create a target column using review ratings.

Rule:
- Rating >= 4 => Good Review
- Rating < 4 => Bad Review

Example:
```python
df['target'] = df['reviews.rating'] < 4
```

### Feature Selection
X:
```python
reviews.text
```

Y:
```python
target
```

### Train Test Split
- Train/Test Ratio = 80/20

## Text Preprocessing

### Tokenization
Use Keras Tokenizer.

Parameters:
```python
MAX_NB_WORDS = 20000
```

Requirements:
- Fit tokenizer on training data.
- Build word index.
- Retain top 20,000 most frequent words.

### Sequence Conversion
Convert:
```python
train_texts -> sequences

test_texts -> sequences
```

Using:
```python
texts_to_sequences()
```

### Padding
Pad all sequences to fixed length.

Parameters:
```python
MAX_SEQUENCE_LENGTH = 150
```

Use:
```python
pad_sequences()
```

### Output Encoding
One-hot encode target classes.

Classes:
- Good Review
- Bad Review

## CNN-LSTM Architecture

### Input Layer
```python
shape=(MAX_SEQUENCE_LENGTH,)
dtype=int32
```

### Embedding Layer
```python
input_dim=MAX_NB_WORDS
output_dim=50
input_length=MAX_SEQUENCE_LENGTH
```

### CNN Block 1
```python
Conv1D(
    filters=64,
    kernel_size=5,
    activation='relu'
)
```

```python
MaxPooling1D(pool_size=5)
```

```python
Dropout(0.2)
```

### CNN Block 2
```python
Conv1D(
    filters=64,
    kernel_size=5,
    activation='relu'
)
```

```python
MaxPooling1D(pool_size=5)
```

```python
Dropout(0.2)
```

### LSTM Layer
```python
LSTM(64)
```

### Output Layer
```python
Dense(
    units=2,
    activation='softmax'
)
```

## Compilation

Optimizer:
```python
Adam
```

Metrics:
```python
accuracy
```

Loss:
```python
categorical_crossentropy
```

## Training

Epochs:
```python
5
```

Batch Size:
```python
64
```

## Evaluation

Evaluate on test dataset.

Calculate:
- Test Loss
- Test Accuracy

Display:
- Training Accuracy
- Validation Accuracy
- Final Test Accuracy

## Task B

Use the complete dataset available from Kaggle.

Objectives:
- Retrain CNN-LSTM model.
- Evaluate on full test dataset.
- Compare performance against subset dataset.

## Deliverables

### Data Preparation
- Dataset loading
- Target creation
- Tokenization
- Sequence conversion
- Sequence padding
- One-hot encoding

### Deep Learning Model
- CNN-LSTM architecture
- Model compilation
- Model training

### Evaluation
- Test loss
- Test accuracy
- Performance comparison

### Final Outcome
Classify customer reviews into:
- Good Review
- Bad Review

using a CNN-LSTM hybrid deep learning architecture.
