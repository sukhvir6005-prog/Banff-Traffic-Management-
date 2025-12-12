**Banff Traffic Management – Smart Mobility & Parking Prediction**

A data-driven machine learning project developed to help the Town of Banff improve parking management, predict congestion, and support real-time mobility decisions. This repository contains notebooks, scripts, visualizations, and deployment files created by Group #3.

**Project Summary**

Banff experiences heavy traffic congestion and limited parking availability, especially during peak tourism months. This project analyzes traffic, parking, weather, and event-related data to:

Identify mobility and congestion patterns

Forecast parking occupancy

Predict near-capacity conditions

Support the development of a real-time mobility dashboard

The final outputs include machine learning models, EDA findings, deployment strategy, and a Streamlit application prototype.

**Repository Structure**

│---Banff-Traffic-Management/

├── ML Canvas Path Finders_3.docx

├── Updated_Models_Banff.ipynb

├── final_banff_parking_model_optimization.py

├── group3_app_rag.ipynb

├── streamlit_app.py

├── visits_dataset_group3.py

├── visualizations_banff_parking_feature_engineering.ipynb

└── .gitignore


**Key Files**

Updated_Models_Banff.ipynb – Model selection, tuning, and evaluation

visualizations_banff_parking_feature_engineering.ipynb – EDA and feature engineering

final_banff_parking_model_optimization.py – Final optimized ML script

streamlit_app.py – Streamlit prediction application

group3_app_rag.ipynb – RAG-based prototype

**Data Collection**

Data covers May to August 2025, representing peak tourism season.

Sources include:

Traffic counters

Road sensors

Camera-based vehicle detection

Weather datasets

Event and tourism schedules

Data metrics include:

Vehicle flow and speed

Parking occupancy and duration

Transit ridership

Pedestrian and trail usage

Weather conditions

**Exploratory Data Analysis (EDA)**

Key Findings:

Traffic peaks between 11 AM and 5 PM

Weekends show higher volume than weekdays

Weather has a measurable effect on mobility patterns

Events cause sudden traffic spikes

August is the busiest month in the dataset

Visualizations include:

Time-series traffic volumes

Weekly and daily heatmaps

Parking occupancy charts

Weather–traffic correlation plots

Parking session analysis

EDA notebook: visualizations_banff_parking_feature_engineering.ipynb

**Machine Learning Models**

Regression Task (Predicting hourly occupancy)

XGBoost Regressor

Mean Absolute Error: 3.961 vehicles

RMSE: 4.435 vehicles

Classification Task (Predicting near-full parking status)

Random Forest Classifier

Accuracy: 99.37 percent

F1 Score: 0.9937

Feature Engineering Techniques:

Time-of-day encoding

Rolling averages

Weather-feature integration

Lag variables

Model notebooks:
Updated_Models_Banff.ipynb
final_banff_parking_model_optimization.py

**Streamlit Application**

A prototype application demonstrating occupancy prediction.

Run locally:

streamlit run streamlit_app.py


The application allows user inputs and returns predicted parking occupancy for selected time periods.

**Deployment Strategy**

Planned deployment approach:

Serialize and save trained machine learning model

Containerize the model API using Docker

Deploy to cloud platforms such as GCP, AWS, or Azure

Integrate the prediction endpoint with a Power BI or web dashboard

Implement monitoring, logging, and accuracy checks

Monitoring plan:

Routine model accuracy audits

Data drift alerts

Quarterly retraining

Sensor data validation

**Challenges and Solutions**

**Challenges:**

Imbalanced data for near-capacity predictions

Limited granularity in hourly weather data

Sensor malfunction outliers

**Solutions:**

Applied advanced imbalance-handling techniques

Integrated higher-frequency weather data

Enhanced time-series modeling with rolling and lag variables

Explored multi-step forecasting for 3–6 hour predictions

**Stakeholder Engagement**

Engagement sessions included:

Presentations to municipal planners

Collaborative sessions with tourism and transportation stakeholders

Identification of high-impact congestion areas

Data requests for event schedules and visitor counts

Stakeholder input guided model development and deployment design.

**Conclusion**

The project successfully analyzed Banff mobility data, built accurate machine learning models, and produced a deployment-ready prediction framework. Results show strong predictive accuracy for both occupancy forecasting and near-capacity classification. Future work includes full integration with municipal dashboards, expanded datasets, and real-time API deployment.
