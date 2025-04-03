# Alphabet Soup Charity Analysis

## Description

This project uses a **deep learning neural network** to classify whether a charity funded by Alphabet Soup will be successful. Leveraging **TensorFlow**, **Pandas**, and **scikit-learn**, the model performs binary classification based on key categorical and numeric indicators.

## Summary

### Part 1: Data Preprocessing

The dataset was cleaned and prepped using standard data science practices.

#### Key Deliverables:

- Load `charity_data.csv` into a **Pandas DataFrame** from the provided cloud URL.
- Drop unnecessary columns: `EIN` and `NAME`.
- Identify the **target** (`IS_SUCCESSFUL`) and **features** (all remaining relevant columns).
- Examine the number of unique values per column.
- Consolidate rare categorical values under an `"Other"` category.
- Encode categorical features using `pd.get_dummies()`.
- Split data into training and testing sets using `train_test_split`.
- Scale feature data using `StandardScaler()` from **scikit-learn**.

### Part 2: Neural Network Model

A deep learning model was built using **TensorFlow/Keras** to classify successful vs. unsuccessful organizations.

#### Key Deliverables:

- Define input dimensions based on the number of features.
- Build a sequential model with:
  - One or two **hidden layers** (ReLU activation).
  - One **output layer** (sigmoid activation).
- Compile the model using **binary crossentropy** and the **Adam** optimizer.
- Train the model on training data, saving weights every 5 epochs using a callback.
- Evaluate the model against test data for **loss and accuracy**.
- Save final model as `AlphabetSoupCharity.h5`.

### Part 3: Model Optimization

Multiple attempts were made to improve accuracy beyond the 75% target.

#### Key Deliverables:

- Adjusted model architecture:
  - Added/removed layers and neurons.
  - Modified activation functions.
  - Tuned the number of training epochs.
- Adjusted data preprocessing:
  - Modified how rare values were grouped.
  - Reconsidered which features to drop.
- Final model saved as `AlphabetSoupCharity_Optimization.h5`.
  
> **Note**: A formal written report was not completed for this project. All findings are documented within the Jupyter notebooks.

## Requirements

### Tools and Libraries

- **Python**
- **Google Colab / Jupyter Notebook**
- **Pandas**
- **scikit-learn**
- **TensorFlow / Keras**
- **HDF5**


pip install pandas scikit-learn tensorflow h5py

## Setup

### Conda Environment

To recreate the working environment for this project:

bash
conda create -n alphabet_env python=3.9
conda activate alphabet_env
pip install pandas scikit-learn tensorflow h5py



# Contact Information

For questions or additional information, reach out to me at:

- **Email:** ilir.hajdari111@gmail.com
