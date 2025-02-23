# Final Project: YouTube Comments Classification - Comparative Analysis of Unsupervised Models

## Overview
This project focuses on classifying YouTube comments into sentiment categories (**Positive**, **Negative**, **Neutral**) using **unsupervised learning** techniques. The goal is to compare the performance of different unsupervised models, such as **Latent Dirichlet Allocation (LDA)**, **Non-Negative Matrix Factorization (NMF)**, and **K-Means Clustering**, in identifying and categorizing sentiments from textual data.

## Project Structure
- **Data**: The dataset contains YouTube comments labeled with sentiments.
- **Notebooks**: Jupyter notebooks for data preprocessing, model training, and evaluation.
- **Models**: Implementation of LDA, NMF, and K-Means for sentiment classification.
- **Visualizations**: Word clouds, t-SNE plots, and confusion matrices to interpret model performance.

## Key Steps
1. **Data Preprocessing**:
   - Clean and preprocess comments (remove punctuation, stopwords, etc.).
   - Vectorize text using **TF-IDF**.

2. **Model Training**:
   - Apply **LDA** and **NMF** for topic modeling.
   - Use **K-Means** for clustering comments into sentiment categories.

3. **Evaluation**:
   - Map predicted topics/clusters to ground truth labels.
   - Compute metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
   - Visualize results using **word clouds** and **t-SNE**.

4. **Comparative Analysis**:
   - Compare the performance of LDA, NMF, and K-Means.
   - Identify strengths and weaknesses of each model.

## Results
- **LDA**: Achieved moderate performance on Positive comments but struggled with Negative and Neutral comments.
- **NMF**: Similar to LDA, with better separation of topics.
- **K-Means**: Provided a different perspective by clustering comments based on similarity.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/cesar1091/cesar1091-Unsupervised_Learning_Final_Project_CSCA5632.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks to preprocess data, train models, and evaluate performance.

## Dependencies
- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `wordcloud`

## Future Work
- Address class imbalance using techniques like **SMOTE**.
- Experiment with advanced text representations like **Word2Vec** or **BERT**.
- Explore supervised learning models for comparison.

## Contributors
- Cesar Aaron Fernandez Niño