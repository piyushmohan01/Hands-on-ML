# Chapter 05 : Support Vector Machines

## Exercises :

- #### **1) What is the fundamental idea behind Support Vector Machines?**

    Linear classifiers can have decision boundaries that sometimes fail to classify the classes or they classify them narrowly, leaving lesser chance for new instances to get classified properly. SVMs on the other hand have decision boundaries that not only seperate the classes properly but also stay far away from the closest training instances as possible, which allows room for new isntances to fall in between the margins.

- #### **2) What is a support vector?**

    Instead of havind one straight line as the decision boundary, SVMs have a street-like boundary with two margins on both sides in addition to the main boundary. Adding instances off the street does not affect the decision boundaries but those that are located on the edge of the streets do and these instances are known as support vectors.

- #### **3) Why is it important to scale the inputs when using SVMs?**

    SVMs are sensitive to feature scales and using unscaled features for SVMs usually result in a stretched and linear street that aligns with one specific axis. This makes it hard to interpret the boundaries and the way they classify new instances and might lead to poor results. Hence, one should always scale the features before using SVMs.
    
- #### **5) Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?**

    Dual form is only faster when the number of features is more than the number of instances. Hence, using the primal form of SVM would be right in this case.

- #### **6) Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ  (gamma)? What about C?**

    In a SVM Classifier with RBF kernel, gamma (γ) acts as a regularization hyperparameter. Increasing both the hyperparameters, γ and C will tackle the case of underfitting.
