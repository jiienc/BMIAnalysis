# BMI Classification

This repository contains code for BMI classification using various machine learning models.

## BMI Dataset

This dataset contains information about individuals' body mass index (BMI) along with their corresponding labels (underweight, normal weight, overweight, and obesity). The dataset is provided in CSV format.

### Files

- `BMI.csv`: The main dataset file in CSV format.

### Columns

The dataset contains the following columns:

- `Height`: Height of the individual in centimeters.
- `Weight`: Weight of the individual in kilograms.
- `BMI`: Body mass index (calculated as weight in kilograms divided by height in meters squared).
- `Label`: BMI label indicating the weight category of the individual (underweight, normal weight, overweight, obesity).

### Usage

The dataset can be used for various purposes, such as:

- Analyzing the relationship between height, weight, and BMI.
- Building machine learning models to predict BMI labels based on height and weight.
- Exploring the distribution of different weight categories in the dataset.

### Acknowledgements

This dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/yasserh/bmidataset), and it was originally contributed by yasserh on Kaggle.

## Installation

To run the code in this project, you need to have Python installed. Additionally, the following Python libraries are required:

- numpy
- pandas
- matplotlib
- seaborn
- pandas_profiling
- scikit-learn

You can install these libraries using the following command:

  ```shell
  pip install numpy pandas matplotlib seaborn pandas_profiling scikit-learn
  ```    

## Usage

1. Clone this repository:

    ```shell
    git clone https://github.com/jiienc/BMIAnalysis.git
    ```

2. Navigate to the project directory:

    ```shell
    cd BMIAnalysis
    ```

3. Open and run the notebook using Jupyter Notebook or JupyterLab. This notebook contains the code for data extraction, exploratory data analysis (EDA), model classification, and evaluation.

4. Follow the instructions in the notebook to execute the code cells and observe the results.

## Results

The notebook compares the performance of several machine learning models for BMI classification. The models evaluated include:

- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC) with linear kernel
- Decision Tree Classifier
- Logistic Regression
- Linear Discriminant Analysis (LDA)

The accuracy, mean squared error, and classification report are provided for each model.
