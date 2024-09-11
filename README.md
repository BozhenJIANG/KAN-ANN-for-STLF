# KAN-NN-for-STLF

## Overview

This documentation provides details on the implementation of a hybrid model combining Kolmogorov-Arnold Networks (KAN) and Artificial Neural Networks (ANN) for interpretable and enhanced short-term load forecasting (STLF). The model is based on a paper published on TechRxiv and can be found by searching for the corresponding article.

## Dataset Source

The dataset used for this implementation can be found at [https://github.com/dafrie/lstm-load-forecasting](https://github.com/dafrie/lstm-load-forecasting). This repository contains electricity load forecasting data for Switzerland, along with related weather and calendar information.

## Environment Setup

To set up the environment for running the KAN-NN-for-STLF model, follow these steps:

1. **Create a new Conda environment**:
   ```bash
   conda create -n kan python=3.9.19
   conda activate kan
   ```

2. **Install required packages**:
   ```bash
   pip install matplotlib==3.6.2 numpy==1.24.4 scikit_learn==1.1.3 setuptools==65.5.0 torch==2.2.2 tqdm==4.66.2 xgboost==2.1.1
   ```

   **Note**: Since the dataset and its processing code originate from the `lstm-load-forecasting` repository, you may also need to install:
   ```bash
   pip install pandas==2.2.2 beautifulsoup4==0.2.0
   ```

   **Note**: The version numbers listed are provided as an example; check for the latest stable versions or compatibility with your system.

## Usage

### Running the Model

To run the model, follow these steps:

1. **Open the Jupyter Notebook**:
   Use the Jupyter Notebook to open the file `"NN+KAN-SPRING.ipynb"`.

2. **Execute the code**:
   Execute the code line by line to run the model and generate the forecast results.

### Attention

- **Reproducing Results**: If you aim to reproduce the results reported in the paper, please ensure not to change any parameters in the model configuration.
- **Exploration and Improvement**: If you wish to explore and potentially improve the model's accuracy, feel free to modify the parameters and experiment with different settings.

## Citation

If the code provided is helpful in your research, kindly cite the original paper:

> Bozhen Jiang, Yidi Wang, Qin Wang, et al. "A Hybrid Kolmogorov-Arnold Networks and Artificial Neural Network based Model for Interpretable and Enhanced Short-term Load Forecasting." TechRxiv, September 6, 2024. DOI: 10.36227/techrxiv.172565795.51369458/v1.

This ensures proper attribution to the authors and promotes the visibility of their research.
