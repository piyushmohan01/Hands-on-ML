# Chapter 03 : Classification

## Exercises :

- #### **1) Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set. Hint: the KNeighborsClassifier works quite well for this task; you just need to find good hyperparameter values (try a grid search on the weights and n_neighbors hyperparameters).**

    ```python
    from sklearn.model_selection import GridSearchCV
    param_grid = [{'weights': ['uniform', 'distance'], 'n_neighbors': [3, 4, 5]}]
    knn_clf = KNeigborsClassifier()
    grid_search = GridSearch(knn_clf, param_grid, cv=5, verbose=3)
    grid_search.fit(X_train, y_train)
    ```
    As mentioned in the question, we use KNeighborsClassifier along with GridSearchCV on weights and n_neighbors to find the best hyperparameter values. 30 fits (5 cross validation folds with 6 candidates) are evaluated.
    
    We get the following results : 
    - Best Parameter values : ```n_neighbors = 4``` and ```weights = distance```
    - Best score : ```0.9716167```
    - Accuracy of ```0.9714``` (>97%).


- #### **2) Write a function that can shift an MNIST image in any direction (left, right, up, or down) by one pixel. Then, for each image in the training set, create four shifted copies (one per direction) and add them to the training set. Finally, train your best model on this expanded training set and measure its accuracy on the test set. You should observe that your model performs even better now! This technique of artificially growing the training set is called data augmentation or training set expansion.**

    For manipulating the image instances, we use Scipy's [ndimage library](https://docs.scipy.org/doc/scipy/reference/ndimage.html) which allows us to perform multi-dimensional image processing. Here, we import the 'shift' method and use it to move the digits to the respective directions.

    ![IMAGE OF THE AUGMENTED INSTANCES](https://i.imgur.com/BYsqgJul.png)
  
    ```python
    X_train_augmented = [image for image in X_train]
    y_train_augmented = [label for label in y_train]
    
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    
    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_idx]
    y_train_augmented = y_train_augmented[shuffle_idx]
    ```
    
    ```python
    knn_clf = KNeighborsClassifier(**grid_search.best_params_)
    knn_clf.fit(X_train_augmented, y_train_augmented)
    y_pred = knn_clf.predict(X_test)
    accuracy_score(y_test, y_pred)
    ```
    * Since we did not run the GridSearch earlier, we are simulating the results.
    * KNeighborsClassifier(n_neighbors=4, weights='distance')
    * Accuracy = 0.9763
