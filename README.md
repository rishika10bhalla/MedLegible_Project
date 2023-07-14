# MedLegible - Interpreting Doctor Notes Using Deep Learning Techniques

## Problem Statement

In the current scenario, Handwriting is also one of the skills to express thoughts, ideas, and language. Over the years, it has been observed that medical doctors have been well known for having illegible cursive handwriting. Therefore, very oftenly it becomes difficult for the patients, pharmacists, and for the researchers to understand the doctor’s notes, Lab reports and prescription. So to make it simple for everyone, such that each and every individual is able to understand and interpret the doctor’s handwriting we will be proposing a model which focuses on accurately predicting and digitizing the cursive handwriting of doctor notes (Lab reports  and prescription) using Deep Convolutional Recurrent Neural Networks.(DCRNN).

##Dataset
The dataset being used IAM Handwriting Database, which is a collection of handwritten text samples. It is commonly used in the field of Optical Character Recognition (OCR) research and contains scanned images of handwritten text along with corresponding ground truth transcriptions. The dataset includes over 13,000 labeled handwriting samples, primarily in English, extracted from forms, letters, and other documents. The samples were collected from different writers and cover various writing styles and qualities.

## Project Overview :
1. **Exploratory Data Analysis:** This step involves analyzing and understanding the data. It includes data visualization, identifying missing values, outliers, and understanding the relationship between the features.
2. **Data Cleaning:** In this step, we handle missing values, outliers, and remove irrelevant features that are not useful in our analysis.
3. **Feature Engineering:** We create new features from existing ones that may improve our model's performance.
4. **Model Training:** Model training using RNN (Recurrent Neural Network) abd Bi-directional LSTM.
5. **Front-end Development:** We create a user-friendly front-end using Streamlit, a Python library for building interactive web applications.

## Deployment :
We deployed the LSTM model using a Streamlit front-end, allowing users to input loan information and receive a prediction of whether the loan will be fully paid or will default.

## Repository Structure:
- data: 
  - Contains the dataset used for the project. 
  - Link for the dataset:
- images:
  - contains three images file for the front end interface.
- src: 
  - FinalMinorCode.ipynb: The ipynb file of the whole code with model training.
  - pyfile.py: the FinalMinorCode ipynb file converted into .py file with name pyfile
  - FRONTEND_FINALFILE.py: The .py file of the streamlit code.
  - handwriting.h5: Contains saved trained model
  - sampleimg2, sampleimg3: Samples which can be given as the input to the trained application.
  - SessionState.py: Contains code to help you create session for your app.  
- models: 
  - Contains the trained model.
    - Bi-directional LSTM
