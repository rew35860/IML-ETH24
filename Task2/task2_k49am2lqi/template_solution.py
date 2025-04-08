# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Additional Imports
# Labeling
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor

# Iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Model Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel

# Cross validation
from sklearn.model_selection import cross_val_score, KFold

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # Handling Categorical (Non-Numeric) Data
    le = LabelEncoder()
    train_df['season'] = le.fit_transform(train_df['season'])
    test_df['season'] = le.transform(test_df['season'])

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    # Imputation
    # Estimate the score after iterative imputation of the missing values
    # with different estimators
    estimators = [
        BayesianRidge(),
        RandomForestRegressor(
            # We tuned the hyperparameters of the RandomForestRegressor to get a good
            # enough predictive performance for a restricted execution time.
            n_estimators=4,
            max_depth=10,
            bootstrap=True,
            max_samples=0.5,
            n_jobs=2,
            random_state=0,
        ),
        KNeighborsRegressor(n_neighbors=15),
        ExtraTreeRegressor()
    ]

    imputer = IterativeImputer(estimator=ExtraTreeRegressor(), max_iter=50, random_state=42, initial_strategy='mean')
    train_df_imputed = imputer.fit_transform(train_df)
    X_test = imputer.fit_transform(test_df)

    # Extracting features and labels
    X_train[:, :2] = train_df_imputed[:, :2]  
    X_train[:, 2:] = train_df_imputed[:, 3:]  
    y_train = train_df_imputed[:, 2]   

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train

        # Define the model
        kernel = self.find_kernel()
        self._model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self._model.predict(X_test)

        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

    def find_kernel(self): 
        best_kernel = DotProduct() + WhiteKernel()
        best_score = float('inf')
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        kernels = [
            DotProduct(),
            RBF(),
            Matern(),
            RationalQuadratic(),
            RationalQuadratic(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)),
            RationalQuadratic() + WhiteKernel(),
            RationalQuadratic(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(),
            DotProduct() + WhiteKernel()
        ]

        for kernel in kernels:
            print(f"Start Training : {kernel}")

            gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
            score = cross_val_score(gpr, self._x_train, self._y_train, cv=kf, scoring='neg_mean_squared_error')
            current_mean = -np.mean(score)

            if current_mean < best_score:
                best_score = current_mean
                best_kernel = kernel
                print(f"New best kernel: {kernel} with mean: {best_score}")

        print(f"Best kernel found: {best_kernel} with mean: {best_score}")
        return best_kernel

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    model = Model()
    # Use this function for training the model
    model.train(X_train=X_train, y_train=y_train)
    # Use this function for inferece
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

