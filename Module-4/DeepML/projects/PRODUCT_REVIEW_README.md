# Product Review Classification Using CNN-LSTM

## Overview

This project implements a Deep Learning based Product Review Classification solution using a hybrid CNN-LSTM architecture built on TensorFlow.

The model analyzes customer review text and classifies reviews as:

- Good Review
- Bad Review

Customer ratings are converted into classification targets where ratings below 4 are treated as negative reviews.

## Dataset

Input file:

```text
GrammarandProductReviews.xlsx
```

Primary columns:

- reviews.text
- reviews.rating

Target generation:

```text
Rating >= 4  -> Good Review (0)
Rating < 4   -> Bad Review (1)
```

## Key Features

- Text preprocessing
- Tokenization and sequence generation
- CNN-LSTM deep learning model
- Binary sentiment classification
- Subset vs Full dataset comparison
- ROC curve analysis
- Confusion matrix visualization
- Automated HTML report generation

## Deep Learning Architecture

```text
Review Text
    ↓
Embedding Layer
    ↓
CNN Layer
    ↓
Max Pooling
    ↓
LSTM Layer
    ↓
Dropout
    ↓
Softmax Output
```

Model Parameters:

- Vocabulary Size: 20,000
- Maximum Sequence Length: 150
- Embedding Dimension: 50
- CNN Filters: 64
- Kernel Size: 5
- LSTM Units: 64
- Dropout: 0.2

## Training Configuration

- Optimizer: Adam
- Learning Rate: 0.001
- Loss: Categorical Crossentropy
- Epochs: 5
- Batch Size: 64
- Early Stopping Enabled
- Learning Rate Scheduler Enabled

## Experiments

### Task A

Training on a subset of the dataset for baseline evaluation.

### Task B

Training on the complete dataset for final evaluation.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Generated Outputs

- Dataset summary
- Training history
- ROC curves
- Confusion matrices
- Model summary
- Experiment comparison report

## Report Output

```text
reports/product_review_cnn_lstm_report.html
```

## Execution

```bash
python product_review_cnn_lstm.py
```

## Project Structure

```text
project/
├── product_review_cnn_lstm.py
├── GrammarandProductReviews.xlsx
├── reports/
│   └── product_review_cnn_lstm_report.html
└── README.md
```
