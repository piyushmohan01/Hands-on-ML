# Chapter 04 : Training Models

## Exercises :

- #### **1) What Linear Regression training algorithm can you use if you have a training set with millions of features?**

    Gradient Descent is suitable for cases such as the one mentioned in the question. It is an optimization algorithm capable of finding solutions to a wide range of problems and works on the idea of tweaking parameters iteratively in order to minimize the cost function. It has different types (Batch, Stochastic and Mini-Batch) for different sub-cases and is relatively faster with large number of features when compared to Normal Equation and SVD methods. [Refer Table 4-1]

- #### **2) Suppose the features in your training set have very different scales. What algorithms might suffer from this, and how? What can you do about it?**

    When the features of the training set have very different scales, the Gradient Descent cost function has the shape of an elongated bowl. This elongation in one or more axes affects the way the algorithm reaches the minimum. If the features were scaled, the path would be simple with steps in nearly the same direction whereas in the case of differently scaled features, the path goes in a direction almost orthogonal to the direction of the global minimum and ends with a long flat line, it eventually reaches the minimum but takes a lot longer. Hence, we should ensure that the features have similar scale and we can do so by using Scikit-Learn's StandardScaler class.

- #### **3) Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?**

    The Logistic Regression cost function is convex, which means that if we pick any two points on the curve, the line segment joining them will never cross the curve. This implies that there are no local minima and just one global minimum and that the slope never changes abruptly. Gradient Descent is guaranteed to find the global minimum, given that the learning rate is not too large and that we wait long enough. Hence, it cannot get stuck in a local minimum when training a Logistic Regression model.
    
- #### **4) Do all Gradient Descent algorithms lead to the same model provided you let them run long enough?**

    All Gradient Descent algorithms have almost no difference after training. They end up with very similar models and make predictions in the exact same way. While training, all three of them reach near the minimum but in different ways and by taking different paths. But eventually, they have similar parameters and a similar cost function, hence a very similar model and predictions. Given that the time taken by the algorithms differ by a margin, we need to let them run long enough to lead to the same model.

- #### **5) Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?**

    If the validation error consistently goes up, it could mean the model is diverging because of a high learning rate. We cannot judge the scenario without bring taking training error into consideration. If the training error also goes up along with the validation error, it indicates the case of diverging which can be fixed by lowering the learning rate and then re-training. If the training error is not increasing along with the validation error, then the model is overfitting and we have to retrain with a different model or different features.
    
- #### **6) Is it a good idea to stop Mini-Batch Gradient Descent immediately when the validation error goes up?**

    With Early stopping regularization, we stop training as soon as the validation error reaches the minimum, i.e. when the validation error stops decreasing and starts to overfit the training data (rise back up). But when using Mini-Batch Gradient Descent, the curves are not smooth and that makes it hard to tell if the model has reached the minimum. Hence, to avoid stopping too early, we stop only after the validation error has been above the minimum for a while and not immediately. Once, we are confident that it will not decrease again, we stop and roll back to the point where it was at minimum.
    
- #### **7) Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?**

    Among the Gradient Descent algorithms mentioned, Mini-Batch GD is the fastest to converge to the vicinity of the optimal solution and is followed by Stochastic GD and then Batch GD. Among the three, Batch GD will actually converge to the optimal solution but it takes a longer time than the others. The others (Mini-Batch and Stochastic GD) can be made to converge with the help of learning schedules that determine the learning rate at each iteration (decreases over time).
    
- #### **8) Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?**

    A large gap between the curves of training error and validation error indicates overfitting of the model on the training set. This can be dealt with in three ways :
    1) **Showing more instances** : One way to improve an overfitting model is to feed it more training instances until the validation error reaches the training error.
    2) **Regularization** : We can regularize an overfitting model by constraining the degrees of freedom. The fewer degrees of freedom it has, the harder it will be for it to overfit the data.
    3) **Early stopping** : Another way to counter overfitting is to stop the learning algorithms as soon as the validation error reaches a minimum. As the epochs go by, the algorithm learns and error goes down but it usually rises back up indicating the start of overfitting.
    
- #### **9) Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?**

    If the training error and validation error curves are almost equal and fairly high, it indicates underfitting of the model. An underfitting model has high-bias and not high-variance. Reducing the regularization hyperparameter α, decreases the model's bias but increases its variance. A good learning rate and ensure the required bias/variance tradeoff to avoid the cases of underfitting and overfitting.

- #### **10) Why would you want to use :**
    - **Ridge Regression instead of plain Linear Regression?**
    - **Lasso Regression instead of Ridge Regression?**
    - **Elastic Net instead of Lasso Regression?**
    
    **i) Ridge over plain Linear Regression** : Ridge Regression adds a regularization term to the general Linear Regression cost function which in turn forces the algorithm to fit the data and also keep model weights small and reasonable. With the help of the additional penalty term, we can avoid the cases of underfitting and overfitting.
    **ii) Lasso over Ridge Regression** : Lasso Regression also adds a regularization term to the cost function but uses the L1 norm instead of the L2 norm used in Ridge Regression. It tends to completely eliminate the weights of the least important features and thus performs feature selection.Hence, Lasso is preferred over Ridge when it seems like only a few features would be useful.
    **iii) Elastic Net over Lasso Regression** : Elastic Net acts as a mix of both Ridge and Lasso Regression and it provides enough regularization along with feature selection. Elastic Net is preferred over Lasso Regression because Lasso Regression sometimes behaves erratically when number of features is more than that of instances or when several features are strongly correlated.
  
- #### **11) Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?**
 
    We would have to implement two Logistic Regression classifiers since an instance (picture) can belong to two different classes at the same time, for example, outdoor and daytime. Softmax Regression predicts only one class at a time, it is multiclass but not multioutput. Hence we can split the four classes as two binary classifiers, i.e, daytime-or-nighttime and outdoor-or-indoor.

