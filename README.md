# Breast Cancer Prediction

This project utilizes machine learning to predict whether a tumor is malignant (0) or benign (1) based on various tumor features such as radius, texture, perimeter, area, and smoothness. We explore different classification models and evaluate their performance.

## Table of Contents
1. [Data Overview](#data-overview)
2. [Data Exploration](#data-exploration)
3. [Data Visualization](#data-visualization)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Neural Network Model](#neural-network-model)
6. [Model Predictions](#model-predictions)
7. [Conclusion](#conclusion)

---

## Data Overview
The dataset consists of 569 entries with 6 columns:
- `mean_radius`: Mean radius of the tumor
- `mean_texture`: Mean texture value of the tumor
- `mean_perimeter`: Mean perimeter of the tumor
- `mean_area`: Mean area of the tumor
- `mean_smoothness`: Mean smoothness value of the tumor
- `diagnosis`: Diagnosis (0 = malignant, 1 = benign)

---

## Data Exploration
- The dataset contains 569 rows and 6 columns.
- The `diagnosis` column is binary, with values `0` (malignant) and `1` (benign).
- Descriptive statistics for the features are calculated to give an overview of the data.
- A correlation matrix is computed, showing strong correlations between `mean_radius`, `mean_perimeter`, and `mean_area`. These features are negatively correlated with the `diagnosis` feature.

---

## Data Visualization
We visualize the distribution of each feature using histograms and the correlation matrix with a heatmap.

### Correlation Heatmap:
A correlation heatmap was plotted to visually understand the relationships between features, confirming that `mean_radius`, `mean_perimeter`, and `mean_area` are strongly correlated with each other, and the diagnosis is negatively correlated with these features.

---

## Model Training and Evaluation
Several machine learning models were implemented to predict breast cancer diagnosis. The models and their respective accuracies are as follows:

1. **K-Nearest Neighbors (KNN)**: Accuracy = 91.23%
2. **Random Forest Classifier**: Accuracy = 92.98%
3. **Decision Tree Classifier**: Accuracy = 88.60%
4. **Support Vector Machine (SVM)**: Accuracy = 87.72%

Among these, the **Random Forest Classifier** achieved the highest accuracy.

---

## Neural Network Model
A neural network was implemented with the following structure:
- **Input Layer**: 5 features
- **Hidden Layers**: Two layers with 512 neurons each
- **Output Layer**: Single neuron with sigmoid activation

Early stopping was used to prevent overfitting, and the training and validation loss improved over time.

---

## Model Predictions
The predictions from the neural network are probabilities between 0 and 1. These can be thresholded (usually at 0.5) to classify the diagnosis as malignant (0) or benign (1).

---

## Conclusion
- The **Random Forest Classifier** performed the best with an accuracy of 92.98%.
- The neural network model showed promising results, with probabilities indicating high confidence in the predictions.
- For more detailed model evaluation, metrics such as precision, recall, F1-score, and ROC-AUC can be computed to assess model performance further.

Feel free to explore the code and models used in this project. The dataset can be found in `Breast_cancer_data.csv`.
