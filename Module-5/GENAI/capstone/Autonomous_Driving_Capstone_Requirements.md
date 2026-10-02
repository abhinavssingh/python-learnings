# Requirement Specification

## Project: Autonomous Driving AI/ML Capstone

## 1. Project Overview
This capstone consists of two independent workstreams:

### Part 1: Vehicle Object Detection
Develop a Deep Learning based object detection solution capable of identifying vehicle types from images and localizing vehicles using bounding boxes. citeturn3search5

### Part 2: Tesla Autopilot Safety Analysis
Perform exploratory data analysis on Tesla Autopilot accident and death records to understand trends, event distribution, and road safety impacts. citeturn3search5

## 2. Business Context
Autonomous Vehicles (AV) and Intelligent Transport Systems (ITS) rely on accurate real-time vehicle detection and safety analytics. The solution should support vehicle classification, localization, and safety trend analysis using historical accident data. citeturn3search5

---

# Part 1 – Object Detection

## Objective
Build a CNN-based object detection model that predicts the vehicle type present in an image and identifies its location using rectangular bounding boxes. citeturn3search5

## Functional Requirements

### FR-1: Dataset Preparation
- Create project folder structure.
- Create child folders for training, validation, testing, and model artifacts.
- Extract and organize Images.zip dataset.
- Prepare images for model training. citeturn3search5

### FR-2: Data Preprocessing
- Validate image integrity.
- Normalize image data.
- Split dataset into train, validation, and test datasets.
- Apply data augmentation where appropriate.

### FR-3: Model Development
- Implement a CNN-based object detection architecture.
- Support vehicle classification.
- Support bounding-box localization.
- Train the object detection model using the prepared dataset. citeturn3search5

### FR-4: Model Evaluation
- Evaluate trained model on test dataset.
- Measure detection accuracy metrics.
- Analyze prediction performance. citeturn3search5

### FR-5: Inference
- Run inference on unseen sample images.
- Visualize predicted bounding boxes.
- Verify vehicle detection quality. citeturn3search5

## Technical Requirements
- Python
- TensorFlow or PyTorch
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Part 2 – Tesla Autopilot Safety Analytics

## Objective
Analyze Tesla Autopilot accident data to understand event distribution, fatalities, model-wise trends, and Autopilot-related safety insights. citeturn3search5

## Functional Requirements

### FR-6: Data Inspection
- Load Tesla-Deaths.csv dataset.
- Inspect schema and data types.
- Identify missing values.
- Remove duplicates.
- Remove irrelevant columns where necessary. citeturn3search5

### FR-7: Exploratory Data Analysis
- Analyze events by date.
- Analyze events by year.
- Analyze events by state.
- Analyze events by country. citeturn3search5

### FR-8: Fatality Analysis
- Calculate deaths per accident.
- Analyze Tesla driver fatalities.
- Analyze occupant fatalities.
- Analyze cyclist and pedestrian fatalities.
- Identify accidents involving multiple victim categories. citeturn3search5

### FR-9: Vehicle Collision Analysis
- Determine frequency of Tesla collisions with other vehicles.
- Analyze collision patterns.
- Generate visual insights. citeturn3search5

### FR-10: Model Analysis
- Analyze event distribution across Tesla models.
- Compare accident frequencies by model. citeturn3search5

### FR-11: Autopilot Verification Analysis
- Analyze verified Tesla Autopilot deaths.
- Compare verified and reported incidents.
- Produce trend visualizations. citeturn3search5

## Technical Requirements
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Non-Functional Requirements
- Reproducible analysis.
- Well-documented code.
- Visualizations for key insights.
- Modular implementation.
- Error handling and data validation.
- Scalable execution for large datasets.

## Deliverables
- Jupyter Notebook for Object Detection.
- Trained Object Detection Model.
- Model Evaluation Results.
- Inference Outputs.
- Jupyter Notebook for Data Analysis.
- EDA Visualizations and Reports.
- Final Capstone Report.
- Requirement Document.

## Acceptance Criteria
- Vehicle detection model trains successfully.
- Model performs object localization using bounding boxes.
- Sample image inference generates accurate detections.
- Tesla accident dataset is cleaned and analyzed.
- Required EDA questions are answered through data visualizations.
- Model-wise and Autopilot-related insights are produced.
- All notebooks execute successfully without errors.
