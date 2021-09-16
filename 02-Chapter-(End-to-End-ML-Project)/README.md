# Chapter 02 : End-to-End ML Project

## Exercises :

- #### **1) Try a Support Vector Machine regressor (sklearn.svm.SVR), with the various hyperparameters such as kernel='linear' (with various values for the C hyperparameter) or kernel='rbf' (with various values fro the C and gamma hyperparameters). Don't worry about what these hyperparameters mean for now. How does the best SVR predictor perform?**

    The SVM Regressor was implemented with the mentioned hyperparameters and the results were compared with those of the other algorithm. Here is the **comparison between the RMSE** when both regressors were fit over the housing_prepared (X_train) and housing_labels (X_test) sets :
    
    * SVM = ```70363.840```
    * Random Forest = ```18603.515```
    * Difference : ```51760.325```
    
    Hence, from these results we can conclude that the Random Forest Regressor works far better than the SVM Regressor in this case. (Both linear and rbf kernels with C values ranging from 10-30000 and 1-1000 respectively and 0.01-3 gamma range for the rbf kernel. 50 candidates - 5 folds. **Best Params : {'C': 30000.0, 'kernel': 'linear'}**)


- #### **2) Try replacing GridSearchCV with RandomizedSearchCV.**

    GridSearchCV was replaced with RandomizedSearchCV which uses the expon and reciprocal functions from scipy.stats package. Here is the **comparison between the RMSE** when both regressors were fit over the housing_prepared (X_train) and housing_labels (X_test) sets :
    
    * SVM + RandomizedSearchCV = ```54767.961```
    * Random Forest = ```18603.515```
    * Difference : ```36164.446```
    
    Hence, from these results we can conclude that the SVM Regressors performs better when the best parameters of RandomizedSearch are taken, instead of GridSearch, though it still shows a greater RMSE when compared to the Random Forest regressor. (Both linear and rbf kernels with C set as the reciprocals with shape parameters 20 and 200000 and a gamma with expon of scale 1.0. **Best Params : {'C': 157055.10989448498, 'gamma': 0.26497040005002437, 'kernel': 'rbf'}**)

- #### **3) Try adding a transformer in the preparation pipeline to select only the most important attributes.**
    
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    
    def indices_of_top_k(arr, k):
        return np.sort(np.argpartition(np.array(arr), -k)[-k:])
        
    class TopFeatureSelector(BaseEstimator, TransformerMixin):
        def __init__(self, feature_importances, k):
            self.feature_importances = feature_importances
            self.k = k
        def fit(self, X, y=None):
            self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
            return self
        def transform(self, X):
            return X[:, self.feature_indices_]
            
    k = 5 # Number of required top features
    preparation_and_feature_selection_pipeline = Pipeline([
        ('preparation', full_pipeline),
        ('feature_selection', TopFeatureSelector(feature_importances, k))
    ])
    
    # Calling fit_transform on the pipeline and storing result
    housing_prep_top_k = preparation_and_feature_selection_pipeline.fit_transform(housing)
    ```
    We have already declared feature_importances from the GridSearch best estimators and the same is being used inside the init function of the class TopFeatureSelector. The fit method calls the user defined function "indices_of_top_k" which as the name says, returns the indices of the k top features. "k" has been declared as 5 here which indicates that we would be getting the transform of 5 top features in the housing_prep_top_k variable once we run the fit_transform method on the pipeline.

- #### **4) Try creating a single pipeline that does the full data preparation plus the final prediction.**

    ```python
    prepare_select_and_predict_pipeline = Pipeline([
        ('preparation', full_pipeline),
        ('feature_selection', TopFeatureSelector(feature_importances, k)),
        ('svm_reg', SVR(**rnd_search.best_params_))
    ])
    
    prepare_select_and_predict_pipeline.fit(housing, housing_labels)
    some_data = housing.iloc[:4]
    some_labels = housing_labels.iloc[:4]
    print("Predictions: ", prepare_select_and_predict_pipeline.predict(some_data))
    print("Labels: ", list(some_labels))
    
    # OUTPUT :
    Predictions: [203214.28978849 371846.88152572 173295.65441612  47328.3970888 ]
    Labels: [286600.0, 340600.0, 196900.0, 46300.0]
    ```
    The pipeline has three diffirent operations :
    *  Running the full_pipeline declared earlier to transform, impute and encode the columns
    * Sorting out the k top features from the dataset and transforming them
    * Using the previous output to run the regressor (SVM) tuned with randomized search
    
    Hence, the pipeline does all tasks from processing to prediction in one single pipeline which is called using the fit method and then used with the predict method to obtain the results.

- #### **5) Automatically explore some preparation options using GridSearchCV.**

    ```python
    param_grid = [{
        'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
        'feature_selection__k': list(range(1, len(feature_importances) + 1))
    }]
    
    grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, 
                                    cv=5, scoring='neg_mean_squared_error', verbose=2)
    grid_search_prep.fit(housing, housing_labels)
    ```
    Here in the param_grid, we have declared a list of strategies for imputing rather than just one like we did earlier. Along with the combinations of strategies we also have the feature_selection_k parameter that has been set to a list of integers ranging from 1 to the number of features or simply the length of our feature_importances list. These different combinations are performed by the GridSearch which other than param grid and Regressor have the same parameters. The result is stored in grid_search_prep which is later fit with the X_train and X_test sets, i.e, housing and housing_labels. The best parameter combination is as shown below :
    ```python
    grid_search_prep.best_params_
    {'feature_selection__k': 15, 'preparation__num__imputer__strategy': 'most_frequent'}
    # Best Imputing strategy : most_frequent
    # Number of Important features : 15 out of 16
    ```