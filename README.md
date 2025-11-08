# ZooAnimalClassifier

Problem Definition and Category
This project focuses on multiclass classification within the zoological domain. The goal is to predict an animal's class in the animal kingdom (Mammal, Bird, Reptile, Fish, Reptile, Amphibian, Bug, Invertebrate) based on its biological and behavioral traits. The classification problem is solved using supervised learning, combining models for improved accuracy and interpretability.
Category: AI/ML Multiclass Classification (Ensemble Learning)

Data Source and Preprocessing Steps
Source: UCI Machine Learning Repository - Zoo Dataset
Features: 16 biologically significant attributes (binary), plus one scaled numerical feature (legs_scaled from the legs feature)
Target: class_type (7 classes of animals)


Preprocessing Summary:
-Converted categorical traits to binary
-Scaled legs to legs_scaled (0–1 range)
-Dropped redundant legs column to avoid duplication
-Removed NaN values from conversion artifacts
-Aligned X_cleaned and y_cleaned for training

 Model Description and Performance Metrics
An ensemble voting classifier was built using:
-Decision Tree: interpretable but prone to overfitting
-K-Nearest Neighbors: local pattern recognition
-Logistic Regression : generalizes well with clean features

Each model was trained individually and compared using:
-Accuracy
-Precision / Recall / F1-score
-Confusion Matrix
-SHAP Feature Importance Visualization
-ROC Curve with OneVsRestClassifier

Reflection: What Worked, What Didn’t, Future Improvements
What Worked:
Ensemble model improved robustness
SHAP visualizations offered clear feature insights
Clean separation of front-end/back-end with Flask API
Feature engineering decisions led to better performance

What Didn’t Work Initially:
Redundant feature (legs and legs_scaled) caused model mismatch
Minor Flask input validation errors
SHAP errors due to data type misalignment

Future Improvements:
Replace manual input fields with drop-downs or toggle switches
Auto-scale legs on frontend from raw value
Include test-time data validation and error prompts
Extend classifier with image-based animal recognition (deep learning)
Deploy as a hosted app with cloud backend and usage logging
