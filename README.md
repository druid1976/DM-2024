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
4. **Run the Classification Test Script:** Run the `ClassificationTest.py` script to perform classification experiment:
    ```bash
    python ClassificationTest.py
    ```
---
5. **Results:** After running the scripts, you'll see various visualizations, outputs, and classification reports indicating the preprocessing steps, feature engineering, feature selection process, and classification performance.
---
6. **Interpretation:** Interpret the results based on the visualizations, outputs, and classification reports provided in the terminal.
---
7. **Optional:** If you want to explore the processed data further or use it for other purposes, uncomment the line at the end of the `PreProcess-FeatureS-FeatureE.py` script to save the processed data to a CSV file named `processed_titanic.csv`. 
- Also you can check the functions and their descriptions which are used inside the `ClassificationTest.py` from `PreProcess-FeatureS-FeatureE.py`.
---
## File Descriptions

- `PreProcess-FeatureS-FeatureE.py`: Python script containing the required fuctions to conduct the experiment.
- `ClassificationTest.py`: Python script containing the classification testing code.
- `train.csv`: Titanic dataset used for training the model.
- `test.csv`: Titanic dataset used for testing the model.
- `processed_titanic.csv` (optional): Processed Titanic dataset saved after preprocessing.
- `Results-Before_and_After-Processing.png`: A picture showing the difference of the result of conducted preprocessing.
---
## Notes

- Ensure that you have the `train.csv` and `test.csv` datasets in the same directory as the scripts.
- The scripts include comments explaining each step for better understanding.
- You can modify the scripts or experiment with different preprocessing techniques, feature engineering strategies, feature selection methods, and classification algorithms to improve the results.
