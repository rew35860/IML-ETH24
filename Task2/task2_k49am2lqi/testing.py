# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Additional Imports 
# Labeling 
from sklearn.preprocessing import LabelEncoder

# Interative imputer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Estimator function 
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor

# Kernel for training 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic

# K Fold 
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

    # Encoding categorical feature - Season 
    lb = LabelEncoder()
    train_df['season'] = lb.fit_transform(train_df['season'])
    test_df['season'] = lb.transform(test_df['season'])


    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    # Init Estimators for imputation : after multiple run, Entree is the best regression estimator 
    estimators = {
        # "baysian_ridge" : BayesianRidge(),
        # "Dtrees" : DecisionTreeRegressor(),
        "Etree" : ExtraTreeRegressor(),
        # "KNreg" : KNeighborsRegressor()
    }

    for estimator in estimators.keys(): 
        # Impute train data 
        imputer = IterativeImputer(estimator=estimators[estimator], random_state=42, initial_strategy='mean', max_iter=50)
        imputer.fit(train_df)
        X_train_temp = imputer.transform(train_df)

        # # Test with randomness to see if the score is fluctuate 
        # np.random.shuffle(X_train_temp)

        # Assign values to the tranning variables
        X_train[:, :2] = X_train_temp[:, :2]
        X_train[:, 2:] = X_train_temp[:, 3:]
        y_train = X_train_temp[:, 2]

        # Impute test data 
        imputer.fit(test_df)
        X_test = imputer.transform(test_df)


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    best_kernel = find_best_kernel(X_train, y_train)
    gpr = GaussianProcessRegressor(kernel=best_kernel, random_state=42)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


def find_best_kernel(X_train, y_train): 
    best_Kernel = DotProduct()
    best_mean = float('inf')
    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    kernels = {
        "DotProduct" : DotProduct(),
        # "RBF" : RBF(length_scale=100.0, length_scale_bounds=(1e-2,1e3)),
        "Matern" : Matern(),
        "RationalQuadratic" : RationalQuadratic()
    }

    # Testing with cross validation on different kernels
    for kernel in kernels.keys(): 
        print(f"Start Training : {kernel}")

        gpr = GaussianProcessRegressor(kernel=kernels[kernel], random_state=42)
        score = cross_val_score(gpr, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        current_mean = -np.mean(score)

        if current_mean < best_mean:
            best_mean = current_mean
            best_Kernel = kernels[kernel]
            print(f"New best kernel: {kernel} with mean: {best_mean}")

    
    return best_Kernel


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results_temp.csv', index=False)
    print("\nResults file successfully generated!")

