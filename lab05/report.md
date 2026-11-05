# Lab 5 Report: Titanic Survival Classification

**Course:** Introduction to Artificial Intelligence (AI-101L)   **Student:** Arham Rizwan

All numbers come from `titanic_classification.ipynb` (stratified 80/20 split, `random_state=42`, 712 training and 179 test passengers).

## EDA findings
- 891 passengers, 38.4% survived, so always predicting "died" scores **61.6%** (61.5% on the test split). This is the baseline.
- Missing values: age 177, embarked 2, deck 688 (77%). Deck was dropped. Age and fare use the training median, embarked the training mode.
- Women survived at 74.2% and men at 18.9%. First class survived at 63.0%, second at 47.3%, third at 24.2%. Cherbourg passengers survived most (55.4%).

## Three-model comparison (default settings, test set)
| Model | Train acc | Test acc | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.804 | 0.783 | 0.681 | 0.729 | 0.859 |
| Decision Tree | 0.983 | 0.782 | 0.727 | 0.696 | 0.711 | 0.758 |
| Random Forest | 0.983 | 0.799 | 0.754 | 0.710 | 0.731 | 0.830 |

The decision tree and random forest memorise the training data (98.3% train against 78 to 80% test). Adding KNN (0.804) and SVC (0.827) in Exercise 4 gives a similar picture.

## Tuning result (GridSearchCV, 5-fold shuffled stratified, training data only)
| Model | Best CV accuracy | Test accuracy | Best parameters |
|---|---|---|---|
| Logistic Regression | 0.8034 | 0.8045 | C = 0.1 |
| Decision Tree | 0.8160 | 0.7765 | max_depth = 4, min_samples_leaf = 10 |
| Random Forest | **0.8231** | **0.8101** | 300 trees, no depth limit, min_samples_leaf = 3 |

## Other results
- Engineered features (`FamilySize`, `IsAlone`): change in test accuracy of 0.0 (LR), -0.034 (DT), -0.005 (RF). They did not help.
- ROC AUC of the tuned Random Forest: 0.836 (`figures/roc_curve.png`).
- Home assignment: a `ColumnTransformer` inside a `Pipeline` gave a best CV accuracy of 0.8216, against 0.8231 for the manual approach, a gap of one passenger. Leakage from fitting the median and scaler before CV exists in the manual approach but is negligible here.

## Deployment recommendation
Deploy the tuned Random Forest (`titanic_best_model.joblib`). It won the cross-validated comparison without the test set being used for selection, and it returns probabilities for the API. The honest caveat is that the models differ by only a few passengers, and logistic regression is nearly as accurate, has a higher default AUC (0.859) and is easier to explain.

## Diabetes (Part A) in one paragraph
Logistic regression on the cleaned diabetes data reaches 72.7% test accuracy (baseline 64.9%), recall 55.6%, ROC AUC 0.817, and 77.1% cross-validated accuracy. Glucose is the strongest predictor. See `diabetes_logistic_regression.ipynb`.

## Files
`titanic_classification.ipynb`, `titanic_best_model.joblib`, `pclass_encoder.pkl`, `gender_encoder.pkl`, `embarked_encoder.pkl`, `preprocessing_info.pkl`, `figures/roc_curve.png`, `diabetes_logistic_regression.ipynb`, `diabetes_model.joblib`, `data/`.
