# California Housing Prices with Support Vector Regression

Comparing RBF and sigmoid kernel Support Vector Regression (SVR) for predicting median house values in California districts.

## Question
How well can SVR predict a district's median house value from census features, and which kernel works better?

## Data
The **California Housing** dataset (1990 census; the version from *Hands-On Machine Learning*) has 20,640 districts. Features:
- longitude and latitude
- housing median age
- total rooms and bedrooms
- population and households
- median income
- ocean proximity

The target is `median_house_value`. The notebook downloads the data automatically.

## Approach
1. **Stratified train/test split** (80/20) on income category, so both sets have the same income distribution.
2. **Preprocessing pipeline:**
   - median imputation and standard scaling for numeric features
   - one-hot encoding for `ocean_proximity`
3. **Hyperparameter search:** `RandomizedSearchCV` (10 iterations, 3-fold CV) over `C` ~ U(1, 11) and `gamma` ~ reciprocal(0.001, 0.1), separately for the RBF and sigmoid kernels.
4. Evaluated the better model on the test set.

## Results
| Model | CV RMSE | Test RMSE |
|---|---|---|
| SVR (RBF) | ~$117,800 | – |
| SVR (sigmoid) | ~$117,200 | **$114,700** |

## How to run
```bash
pip install numpy pandas scikit-learn scipy
jupyter notebook "SVM Regression on California housing dataset.ipynb"
```

## Limitations and next steps
- A test RMSE of ~$115k is about the same as the standard deviation of house values, so the model is roughly as good as **predicting the mean**. The cause is that the target (values up to $500k) isn't scaled, while the search uses small `C` values and the default `epsilon=0.1`. Scaling the target (e.g. `TransformedTargetRegressor`) or searching much larger `C` values should improve this dramatically.
- Both kernel searches use the same random seed, so they tried identical parameters. That's fine, but worth stating.
- Add a baseline (mean predictor, linear regression) so the results have context.
- Fit on a subsample or use `LinearSVR`, since SVR scales poorly beyond ~10k rows.
