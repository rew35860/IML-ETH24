# 1. Task description
# Ridge Regression with Cross-Validation

## Overview
This task is about using **cross-validation** for **ridge regression**. Remember that ridge regression can be formulated as the following optimization problem:

```math
\min_w \sum_{i=1}^{n} (y_i - w^T x_i)^2 + \lambda ||w||_2^2
```

where:
- \( y_i \) is the label of the \( i^{th} \) data point,
- \( x_i \) is the feature vector for the \( i^{th} \) data point,
- \( \lambda \) is a hyperparameter controlling the regularization strength.

## Regularization Parameters
We will evaluate ridge regression using the following values of \( \lambda \):

```math
\lambda_1 = 0.1, \quad \lambda_2 = 1, \quad \lambda_3 = 10, \quad \lambda_4 = 100, \quad \lambda_5 = 200
```

## Task Description
1. Perform **10-fold cross-validation** for ridge regression on each \( \lambda \) value.
2. Train ridge regression **10 times**, leaving out a different fold each time.
3. Compute and report the **average Root Mean Squared Error (RMSE)** over the 10 test folds.

## Root Mean Squared Error (RMSE)
The RMSE is calculated as follows:

```math
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
```

where:
- \( y_i \) are the actual ground truth labels,
- \( \hat{y}_i \) are the predicted labels from the ridge regression model.

## Guidelines
- Use **scikit-learn** for ridge regression and cross-validation.
- No feature transformation or scaling should be performed.
- Assume that each data point is **independently and identically distributed (iid)**.
- Refer to **Section 3.1.2.1** of the scikit-learn user guide for more details on K-fold cross-validation.

## Objective
The goal of this task is to test your understanding of **ridge regression with cross-validation** by implementing the described methodology accurately.

---

### References
- [Scikit-learn User Guide: Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Scikit-learn User Guide: Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)