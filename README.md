# Titanic Survival Prediction Experiment

This experiment aims to predict survival on the Titanic using machine learning techniques. The experiment involves data preprocessing, feature engineering, feature selection, and model training.

## Prerequisites

- Python 3.x
- Required Python libraries: pandas, numpy, scikit-learn, seaborn, matplotlib

## Steps to Reproduce the Experiment

1. **Clone the Repository:** Clone this repository to your local machine.
```bash 
    git clone https://github.com/druid1976/DM-2024.git
```
---
2. **Navigate to the Directory:** Open your terminal and navigate to the directory where the repository is cloned.
---
3. **Install Dependencies:** If you haven't installed the required Python libraries, install them using pip:
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib
    ```
---
4. **Run the Experiment Script:** Run the script `PreProcess-FeatureS-FeatureE.py` using Python:
    ```bash
    python PreProcess-FeatureS-FeatureE.py
    ```
---
5. **Run the Classification Test Script:** Optionally, you can run the `ClassificationTest.py` script to perform classification testing:
    ```bash
    python ClassificationTest.py
    ```
---
6. **Results:** After running the scripts, you'll see various visualizations, outputs, and classification reports indicating the preprocessing steps, feature engineering, feature selection process, and classification performance.
---
7. **Interpretation:** Interpret the results based on the visualizations, outputs, and classification reports provided in the terminal.
---
8. **Optional:** If you want to explore the processed data further or use it for other purposes, uncomment the line at the end of the `PreProcess-FeatureS-FeatureE.py` script to save the processed data to a CSV file named `processed_titanic.csv`.
---
## File Descriptions

- `PreProcess-FeatureS-FeatureE.py`: Python script containing the main experiment code.
- `ClassificationTest.py`: Python script containing the classification testing code.
- `train.csv`: Titanic dataset used for training the model.
- `test.csv`: Titanic dataset used for testing the model.
- `processed_titanic.csv` (optional): Processed Titanic dataset saved after preprocessing.
---
## Notes

- Ensure that you have the `train.csv` and `test.csv` datasets in the same directory as the scripts.
- The scripts include comments explaining each step for better understanding.
- You can modify the scripts or experiment with different preprocessing techniques, feature engineering strategies, feature selection methods, and classification algorithms to improve the results.
