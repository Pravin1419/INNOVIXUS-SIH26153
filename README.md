# INNOVIXUS – Predictive Cyber Defence Using Temporal World Models

## Overview

INNOVIXUS is an AI-based predictive cyber-defence prototype that learns temporal network behaviour and forecasts future network states using an LSTM-based World Model.

Instead of analysing network traffic only as individual events, the system models how network states evolve over time and estimates possible attacker progression.

## Key Features

- Temporal network-state modelling
- LSTM World Model
- K-step future-state forecasting
- Infiltration probability estimation
- MITRE ATT&CK contextual mapping
- SHAP-based explainability
- PCAP packet-feature extraction
- Interactive Streamlit dashboard
- Logistic Regression baseline

## System Architecture

CSV / PCAP  
↓  
Feature Extraction  
↓  
Feature Fusion & Preprocessing  
↓  
10-Second Network States  
↓  
Temporal Sequence Builder  
↓  
LSTM World Model  
↓  
K-Step Future-State Forecasting  
↓  
Infiltration Prediction  
↓  
MITRE ATT&CK Context + SHAP  
↓  
Streamlit Dashboard

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Scapy
- Streamlit

## Dataset

CSE-CIC-IDS2018

The current model evaluation uses selected CIC-IDS2018 flow telemetry from 14 February 2018.

## Project Files

- `preprocess.py` – preprocesses network-flow data
- `build_states.py` – builds time-based network states
- `create_sequences.py` – creates temporal sequences
- `train_world_model.py` – trains the LSTM World Model
- `forecast.py` – performs K-step future forecasting
- `train_infiltration_classifier.py` – trains the infiltration classifier
- `explain_prediction.py` – generates SHAP explanations
- `mitre_mapper.py` – provides contextual MITRE mapping
- `pcap_parser.py` – extracts packet-level features from PCAP
- `feature_fusion.py` – combines flow and packet features
- `benchmark_temporal.py` – performs temporal benchmarking
- `analyze_states.py` – analyses network states
- `dashboard.py` – Streamlit demonstration dashboard

## Installation

Create and activate a Python virtual environment, then install:

```bash
pip install -r requirements.txt