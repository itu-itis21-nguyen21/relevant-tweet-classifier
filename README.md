# AI-Powered Turkish Tweet Classification

## Project Overview

This project was developed during my internship with the goal of designing and implementing an AI solution to predict whether a Turkish tweet can be utilized by one of my clients. To achieve this, I framed the problem as a classification task with three categories:

- **Irrelevant**
- **Client A**
- **Client B**

## Approach

To conduct the classification, I designed a **Multilayer Perceptron (MLP)** with the following specifications:

- **Input Features:** 768-dimensional vector embeddings extracted from each tweet using a **BERT-based sentence embedding model** trained specifically for Turkish.
- **Architecture:** Fully connected layers with **LeakyReLU** activation and **Dropout** for regularization.
- **Optimizer:** **AdamW** optimizer with:
  - Learning rate: **2e-5**
  - Weight decay: **1e-2** (default)

## Performance

Below is the classification report of the best-performing model on the test set:

```
!(report.png)
```

## Repository Structure

```
📂 project-root
│── 📁 model                                                 # Model implementation
│──── preprocess_and_split_train_and_test_sets.py            # Script for preprocessing data and splitting data into train and test sets
│──── feature-extraction.py                                  # Script for extracting vector embeddings from input tweets
│──── BertRelevanceClassifier.py                             # Script for defining the MLP
│──── MulticlassTrainer.py                                   # Script for training the model
│──── MulticlassTester.py                                    # Script for testing the model
│── README.md              # Project documentation
```

