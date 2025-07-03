
# Detecting Misinformation on Social Media Using NLP + Transformers

## Overview

This project applies advanced NLP techniques to detect misinformation in social media content. We fine-tune transformer-based models (BERT, RoBERTa) on the LIAR dataset and compare them to traditional machine learning baselines (Logistic Regression with TF-IDF and LSTM). The goal is to classify political statements into six credibility labels (true, mostly true, half true, mostly false, false, pants-on-fire) and assess the effectiveness of modern deep learning approaches for fake news detection.

## Project Directory Structure

```
LIAR-Detection/
├── data/
├── utils/
├── 01_feature_enginering.ipynb
├── 02_exploratory_data_analysis.ipynb
├── 03_logistic_regression.ipynb
├── 04_lstm.ipynb
├── 05_bert.ipynb
├── 06_roberta.ipynb
├── README.md
├── requirements.txt
└── Project Presentation.pdf
```

## How to Run the Code

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0
- Transformers (HuggingFace)
- scikit-learn
- pandas
- matplotlib, seaborn
- tqdm

To install required packages:

```bash
pip install -r requirements.txt
```

### Data Preparation

- The required CSV files (`train.csv`, `validation.csv`, `test.csv`) are provided in the `data/` folder.
- These files are preprocessed versions of the LIAR dataset, ready for use in all notebooks.

### Running the Notebooks

Run the following notebooks in order for reproducibility:

1. `01_feature_enginering.ipynb`
2. `02_exploratory_data_analysis.ipynb`  
3. `03_logistic_regression.ipynb`  
4. `04_lstm.ipynb`  
5. `05_bert.ipynb`  
6. `06_roberta.ipynb`  

**Note:** GPU acceleration is recommended for training BERT and RoBERTa.

## How to Engage with the Project

- All code is in Jupyter notebooks and is designed to be self-contained and reproducible.
- You can adjust hyperparameters (batch size, learning rate, epochs) in the training sections of each notebook.
- The trained models can be further saved or exported for deployment.
- The evaluation section in each notebook provides metrics, confusion matrices, and performance visualizations.

## Challenges & Limitations

### Challenges Experienced

- **Data Imbalance:** Certain labels such as *pants-on-fire* are under-represented, making it difficult to achieve high recall for rare classes.
- **Metadata Usage:** While initial experiments focused on text alone, integrating metadata (speaker, party affiliation, etc.) remains an open challenge for improving model performance.
- **Compute Requirements:** Fine-tuning large transformer models is resource-intensive. Limited GPU time restricted extensive hyperparameter sweeps.

### Upcoming Challenges

- **Generalization:** Current models are trained on political statements. Extending them to broader social media content may require domain adaptation.
- **Explainability:** Transformer models are often black boxes. Developing interpretability techniques to understand why a statement is classified as fake/true is a desirable future direction.
- **Multi-modal Inputs:** The paper we reviewed suggests combining textual, visual, and social signals for fake news detection — this project currently focuses on text alone.

### Limitations

- The project uses the LIAR dataset only; thus, performance may not generalize to all social media platforms or other types of misinformation.
- The absence of additional propagation or user features limits the model's ability to exploit social context for fake news detection.

## Summary

This project provides a foundation for applying modern NLP techniques to misinformation detection. The code, data, and notebooks are organized and documented for ease of continuation. Future development can build upon this work by incorporating multi-modal features, improving class balance handling, and expanding model generalization.

## Project Team

- Arabind Meher  
- Uday Pothuri  
- Alan Uthuppan 
- Manan Patel
