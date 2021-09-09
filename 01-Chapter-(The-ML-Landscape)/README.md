
# Chapter 01 : The Machine Learning Landscape

## Exercises :

- #### **1) How would you define Machine Learning? (P04)**

    > Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.
    > - *Arthur Samuel, 1959*
    
    > A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
    > - *Tom Mitchell, 1997*
    
- #### **2) Can you name four types of problems where it shines? (P07)**

1) Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning Algorithm can often simplify code and perform better.
2) Complex problems for which there is no good solution using a traditional approach: the best Machine Learning techniques can find a solution.
3) Fluctuating environments: a Machine Learning system can adapt to new data.
4) Getting insights about complex and large amounts of data.

- #### **3) What is a labeled training set? (P08)**

    A labeled training set is one that has pre-defined labels/results for all its training examples. Usually, such sets are used within the "Supervised" learning category of Machine Learning wherein, in order to train our model, we provide it with examples along with their right answers/labels. For example, the training set containing email patterns, keywords, etc. and their corresponding categories of "Spam/Ham" labeled with them (or) the Titanic Train Dataset that has its features with their right outcomes (here, Survived/Dead).
    
- #### **4) What are the two most common Supervised tasks? (P09)**

    The two most common tasks under Supervised Learning are as follows :
    * **Regression** : Here, the task is to predict a *target* numeric value when provided a set of features called *predictors* using some learning parameters. A common example would be the task of Predicting the House Prices (*target*) based on the area, number of rooms, floorsm etc. (*predictors*)
    
    * **Classification** : Here, as the name says, we classify/group the data based on the similarities between the training examples and then using these classifications, the model predicts which group would the new example fall in. A good example would be the classification of Spam/Ham emails where we have two seperate outcomes/groups.
    
- #### **5) Can you name four common Unsupervised tasks? (P12-P13)**

    The four common tasks under Unsupervised Learning are as follows :
    * **Dimensionality Reduction** : Task where the goal is to simplify the data without losing too much information. This helps in making the algorithm run faster and more efficient, both in memory and performance. (Merge correlated features into one)
    
    * **Anomaly Detection** : Here, the system is shown mostly normal instances during training, so it learns to recognize them and when it sees a new instance it can tell whether it looks like an anomaly or not. For example, the Detection of Unusual Card Activity to prevent fraud (Catching manifacturing defects or automatically removing outliers from the dataset before training)
    
    * **Novelty Detection** : This is very similar to Anomaly Detection but the only difference is that here the algorithms expect to see only normal data during training, while Anamoly Detection algorithms are usually more tolerant and perform well even with a small percentage of outliers while training.
    
    * **Association Rule Learning** : Here the goal is to dig into large amounts of data and discover interesting relations between attributes.
    
- #### **6) What type of Machine Learning algorithm would you use to allow a robot walk in various unknown terrains? (P14-P15)**

    Reinforcement Learning Algorithms can be used to train such robots, where we would set a reward/penalty system (For every right action it would get a reward and for every wrong action, a penalty) This way by analyzing several terrains and trying to walk on them (*action*), the robot (*agent*) learns the best reward-winning policy and performs accordingly to new unknown terrains.

- #### **7) What type of algorithm would you use to segment your customers into multiple groups? (P11)**

    A Clustering Algorithm (Unsupervised) can be used for this purpose, wherein, we provide the customer data to the model and it detects the underlying patterns and similarities between several customers based on certain featutes and forms smaller groups within the entire training dataset. This might help understand the target audiences and their preferences in a better way and hence, improve the reach (or/and) services.

- #### **8) Would you frame the problem of Spam Detection as a Supervised learning problem or an Unsupervised learning problem? (P08)**

    The Spam Detection problem can be viewed as a Supervised learning problem or in more detail *Supervised-Online-Model-Based* learning problem where the labels corresponding to the training examples are provided to the model (*Supervised*) and it rapidly adapts to changing data (*Online/Incremental*) preparing itself and creating a model on the given dataset and then predicting the outcome on the unknown data (*Model-Based*)

- #### **9) What is online learning system? (P16)**

    In online learning, we train the system *incrementally* by feeding it data instances sequentially, either individually or by small groups called *mini-batches*. Each step is fast and cheap, so the system can learn about new data on the fly, as it arrives, hence adapting to changes rapidly (*the learning rate parameter*). The big challenge faced by this type of learning is that if bad data is provided to the system, it learns about it incrementally and the performance gradually declines.
    
- #### **10) What is out-of-core learning? P(17)**

    Online Learning Algorithms can also be used to train systems on huge datasets that cannot fit in one machine's main memory. The algorithm loads part of the data, runs a training step on it and repeats the process until it runs out of all the data available. This method is called out-of-core learning.
    
- #### **11) What type of learning algorithm relies on a similarity measure to make predictions? (P18)**

    The Instance-Based learning, the system learns the examples by heart and then generalizes to new cases by comparing them to the learned examples using a similarity measure. The similarity measure acts as the key component for the instance-based learning systems to predict the outcomes.
    
- #### **12) What is the difference between a model parameter and a learning algorithm's hyperparameter? (P20-P30)**

    The differences between Model Parameters and Algorithm Hyperparameters are as follows:
    - **Model Parameter** : The parameters used to represent the model being used and ones that can be tweaked represent different forms of the model are called Model Parameters. By convention "Θ" (*theta*) is used to represent them.

    - **Algorithm Hyperparameters** : The parameters of a learning algorithm and not a model, those that are not affected by the learning algorithm itself and must be set prior to training and remain constant during the training are called Hyperparameters. Hyperparameter tuning is an important part of buildin Machine Learning systems.
    
- #### **13) What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make precictions? (P21)**

    In Model-Based learning algorithms, the common strategy/process is as follows : 
    * First the model parameters are defined. (Θ)
    * Then, in order to find the optimal values for these parameters, the system is provided with a performance measure (*utility function/fitness function*) that measures how *good* the model performs or a cost function to measure how *bad* it performs with the data.
    * This process of *model training* is where the algorithm searches for the optimal parameter values for the model and then goes on to make optimal predictions with the same.
    
- #### **14) Can you name four of the main challenges in Macine Learning? (P24-P30)**

    1) **Insufficient Quantity of Data** : It takes a lot of data for most Machine Learning algorithms to work properly.
    2) **Non-representative Data** : In order to generalize well, it is crucial that training data be representative of the new instances it will generalize to. (or adding new instances does not significantly affect the overall performance) This also gives rise to *sampling bias* which means that if the data is too small, it will have *sampling noise* and on the other hand if its too large and the sampling method is flawed, it would again result in non-representative data.
    3) **Poor-Quality Data** : If the training data is full of errors, outliers and noise, it makes it harder for the system to detect the underlying patterns and hence it makes it less likely to perform well.
    4) **Irrelevant Features** : The system is capable of learning only when the data contains enough relevant features and the process of *feature engineering* tries to form a good set of relevant features for the model. This process involves : feature selection, feature extraction and creating new features.
    5) **Overfitting the Training Data** : The case where the model performs very well on the training data but fails to generalize new unknown data is known as overfitting. (Overanalyzing the patterns of the training set does not help in generalized predictions)
    6) **Underfitting the Training Data** : This is the opposite case of overfitting where the model fails to detect the underlying patterns in the provided training set. (Few ways to tackle these two cases are : Regularization, Feature engineering, Hyperparameter Tuning, etc.)
    
- #### **15) If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions? (P29)**

    This is the case of Overfitting (described above) and here are three solutions for the same :
    
    1) To simplify the model by selecting one with fewer parameters, by reducing the number of attributes in the training data or by constraining the model. (Regularization)
    2) To gather more training data.
    3) To reduce the noise in the training data. (Data cleaning)
    
- #### **16) What is a test set and why would you want to use it? (P31)**

    For testing and validating our models, we usually split the data into two halves : train set (80%) and test set (20%). We train our model on the train set and then test it over the test set. By evaluating the model on the test set, we get the estimate of training error (error between test and train set) which is later compared to the generalization error (error when unknown new instances are provided). If the training error is low and the generalization error is high, the model is overfitting the data.
    
- #### **17) What is the purpose of validation set? (P32)**

    One solution to the problem of overfitting with regularization is *holdout validation* where we hold out part of the training set to evaluate several candidate models and then select the best one. This new heldout set is called *validation set* and is also referred as development set. The validation set should not be too small or too large when evaluating models.
    
- #### **18) What can go wrong if you tune hyperparameters using the test set? (P32)**

    When comparing between various models with their respective hyperparameter values over the test set, we measure the generalization error multiple times on the same set and then we fine tune the hyperparameters to produce the best results for that particular set. This makes the model work great on that set but not on new data. (This way of picking the hyperparameter valuse only favours that one particular set)
    
- #### **19) What is repeated cross-validation and why would you prefer it to using a single validation set? (P32)**

    A very small or very large validation set is not ideal for evaluation and might lead to selection of suboptimal models. One solution to this problem is to perform repeated *cross validation*, using several small validation sets instead of one. Each model is evaluated once per validation set, after it is trained on the rest of the data. By averaging out all the evaluations of a model, we get a much more accurate measure of its performance.
