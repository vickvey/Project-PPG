# Project-PPG Documentation

**Author:** [Vivek Kumar](https://github.com/vickvey)

This document provides a comprehensive overview of the end-to-end machine learning (ML) pipeline developed for the PPG Project. 

It details each stage of the process, from data extraction and preprocessing to model training and evaluation, ensuring clarity and reproducibility for future reference.

Here is the quick glance of the ML pipeline:  
![Pipeline Diagram](ViewPipeline/pipeline.drawio.png)

## Table of Contents
1. [Introduction](#introduction)
2. [PPG Overview](#ppg-overview)
3. [HRV Analysis](#hrv-analysis)
4. [Data Collection](#data-collection)
5. [Data Extraction](#data-extraction)
6. [Data Preprocessing](#data-preprocessing)
7. [Feature Reduction](#feature-reduction)
8. [Model Evaluation](#model-evaluation)
9. [Results](#results)
10. [Conclusion](#conclusion)

## Introduction
The PPG Project aims to develop a machine learning pipeline for analyzing physiological data to detect anxiety levels. 

This documentation outlines the workflow, from raw data collection to model evaluation, providing detailed explanations for each step and the scripts involved.

## PPG Overview
### What is PPG?
Photoplethysmography (PPG) is a non-invasive optical method to measure blood volume changes in the skin. It monitors heart rate and blood oxygen (SpO2) levels.

### How PPG Works
1. **Light Emission:** An LED shines light into the skin.
2. **Absorption and Reflection:** Blood absorbs light, and the rest is reflected back.
3. **Detection:** A sensor measures the reflected light.
4. **Blood Volume Changes:** With each heartbeat, blood volume changes, altering light absorption.
5. **Signal Generation:** The sensor converts light changes into an electrical signal (PPG waveform).
6. **Data Analysis:** The PPG signal is analyzed to determine heart rate and blood oxygen levels (SpO2).

### Key Points
- **Heart Rate Monitoring:** Detects heartbeats to calculate heart rate.
- **SpO2 Monitoring:** Uses dual-wavelength light to estimate blood oxygen levels.
- **Non-Invasive:** Suitable for continuous use in wearables like fitness trackers and smartwatches.

### Applications of PPG
1. Heart Rate Monitoring
2. SpO2 Monitoring
3. HRV Analysis
4. Sleep Monitoring
5. Fitness Tracking
6. Cardiovascular Detection
7. Stress Monitoring
8. Remote Patient Monitoring
9. Blood Pressure Estimation
10. Peripheral Circulation Monitoring
11. Medical Research

## HRV Analysis
### What is HRV?
Heart Rate Variability (HRV) of a PPG (Photoplethysmogram) signal is the measurement of the variation in time intervals between successive pulse peaks reflecting the heartbeats. HRV provides insights into autonomic nervous system function and cardiovascular health.

### Why Measure HRV?
1. **Cardiovascular Health:** Higher HRV indicates better cardiovascular health and resilience.
2. **Stress and Recovery Monitoring:** Tracks stress levels and recovery in everyday life and athletic training.
3. **Mental Health Monitoring:** Lower HRV is associated with anxiety, depression, and other mental health conditions.
4. **Early Health Warnings:** Changes in HRV can provide early warnings of potential health issues.
5. **Chronic Condition Management:** Useful in managing conditions like diabetes, hypertension, and heart disease.
6. **Research and Clinical Studies:** Widely used to study the ANS and its impact on various physiological and pathological conditions.

### HRV Metrics
Here are some examples of HRV metrics which one can get from any body signal.
#### Time-Domain HRV Metrics
- **MeanNN:** The average length of the intervals between normal heartbeats.
- **SDNN:** Standard deviation of the NN intervals.
- **RMSSD:** Square root of the mean squared differences between successive NN intervals.
- **pNN50:** The proportion of successive NN intervals that differ by more than 50 milliseconds.

#### Frequency-Domain HRV Metrics
- **LF (Low Frequency):** Represents the power in the low-frequency range (0.04–0.15 Hz).
- **HF (High Frequency):** Represents the power in the high-frequency range (0.15–0.4 Hz).
- **LF/HF Ratio:** Provides insights into the balance between sympathetic and parasympathetic activities.

## Data Collection

We conducted a survey to collect PPG data from 101 participants, capturing their PPG time series data during various activities. This project specifically uses `activity1` data from all participants.

### Raw Data
- **Directory:** `Raw`
  - **Subdirectory:** `GSR_data`
    - **Description:** Contains directories for each participant (e.g., `101_GSR`, `102_GSR`), with each directory holding CSV files for different activities. For this project, we are using the `activity.csv` file from each participant's directory.
  - **File:** `participant_response.xlsx`
    - **Description:** Contains self-assessment anxiety test data collected from participants.


## Data Extraction
### Scripts
- **Feature Extraction:**
  - **Script:** `FeatureExtraction.py`
  - **Description:** Extracts relevant features from the raw data.
- **Label Extraction:**
  - **Script:** `LabelExtraction.py`
  - **Description:** Extracts corresponding labels for the features.

### Output
- **Raw Features and Labels:**
  - **Files:** `features.csv`, `labels.csv`
  - **Description:** CSV files containing extracted features and labels from the raw data.

### Merging Features and Labels
- **Script:** `MergingData.py`
- **Output File:** `merged.csv`
- **Description:** Merges the extracted features and labels into a single dataset for analysis.

## Data Preprocessing
### Initial Data
- **File:** `merged.csv`
- **Description:** Combined dataset containing both features and labels.

### Feature Removal I
- **Script:** `FRbyMissing.py`
- **Output File:** `reduced_I.csv`
- **Description:** Removes features based on missing values.

### Correlation Plotting (85 Features)
- **Files:**
  - `corr_mat_I.xlsx`
  - `corr_hmap_I.png`
- **Script:** `FRbyCorr.py`
- **Description:** Plots correlation matrix for initial 85 features.

## Feature Reduction
### Feature Removal II
- **Script:** `FRbyCorr.py`
- **Output File:** `reduced_II.csv`
- **Description:** Further reduces features based on correlation analysis.

### Correlation Plotting (41 Features)
- **Files:**
  - `corr_mat_II.xlsx`
  - `corr_hmap_II.png`
- **Script:** `FRbyCorr.py`
- **Description:** Plots correlation matrix for the reduced set of 41 features.

## Model Evaluation
- **Scripts:**
  - Various models with KNN Imputer and Robust Scaler
  - Hyperparameter tuning scripts
- **Description:** Evaluates different models on the preprocessed data with hyperparameter tuning to identify the best-performing model.
- **Output:** `model_comparison.pdf`
- **Description:** PDF file containing the comparison of different models' performance.

## Results
- **File:** `model_comparison.pdf`
- **Description:** Document summarizing the performance of different models, including metrics and visualizations.

## Conclusion
This documentation provides a detailed walkthrough of the ML pipeline for the PPG Project, from raw data extraction to model evaluation. By following this guide, the entire process can be reproduced and validated, ensuring transparency and reliability in the analysis.
