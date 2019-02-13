# ALLSTATE-CLAIMS-SEVERITY
# INTRODUCTION
Allstate is a personal insurance company in United States. They protect almost 16 million households and want to improve their service to them. When an insurance is claimed there is a lot of paper work to be done which is a waste of time and energy. Allstate is developing methods for automatic prediction of the cost which indicates the severity of the claim to provide better customer prediction. So our main aim here is to develop an algorithm to predict the cost of an accident and then predicting the severity of the insurance claim based on it.
# PROBLEM DESCRIPTION
This project is a Kaggle Competition. The problem statement involves a challenge of predicting the cost value based on a set of parameters using Big data analytics and Machine learning algorithms. In order to predict the cost, we used various regression algorithms(linear regression, gradient boosting, random forest) using SparkML in Python(Pyspark) with spark cluster. The competition evaluation metric is mean absolute error between predicted loss and actual loss.
# DATASET DESCRIPTION
The dataset for this project provided by Allstate is obtained from
https://www.kaggle.com/c/allstate-claims-severity/data.
The dataset consists of two CSV files namely, train and test dataset. Each row in these datasets represents an insurance claim. The ‘loss’ column of the test dataset is to be predicted.
    
The training data has total 188319 instances. Every instance has 132 fields which includes id, categorical values, continuous values & the label ‘loss’. Field names have prefixes ‘cont’ and ‘cat’ which indicate the type of values they hold. Cat indicates categorical values and ‘cont’ indicates continuous variable. There are no missing values. The column named ‘loss’ in the train dataset is the cost implying the severity of the claim.
The following are the features of the datasets:
➢ Categorical features are labelled as ‘a’, ‘b’, ‘as’ or ‘sz’.
➢ Continuous features are float values varying within 1.
➢ Number of features are 130 with 116 Categorical and 14 Continuous features – excluding
‘id’ & target label ‘loss’
➢ Output variable ‘loss’ is a double value.
➢ Number of records in train dataset = 188319
➢ Number of records in test dataset = 125547
There is no test dataset given, so we randomly split the training dataset into two datasets. One for training and the other for validation.
# EXPERIMENTAL METHODOLOGY
The approach we used for this project is as follows:
• Pre-processing of Data
▪ Converting categorical data to numeric data
▪ Feature Extraction
• Training the dataset
▪ Implemented Regressor on the data
▪ Found the best set of parameters for which the model performed best
• Techniques Evaluation
▪ We evaluated the regressor models using the flowing three metrics - Mean Absolute Error, Root Mean square error and R square.
# PRE-PROCESSING TECHNIQUES
As the dataset given is very huge, it is a good practice to pre-process the data (as useful data gives better results). With the increase in input features, it becomes harder to visualize the dataset and sometimes the features are redundant and correlated and hence we reduce the number of dimensions. In the process of dimensionality reduction, we obtain a set of principle variables from a large set of input variables while keeping the relevant information intact required by the predictive model to produce accurate results. The data is first loaded using pandas read function. Using the pandas read function, we can get the first six records of all the variables. The results obtained are as follows:
  
 The type of the variables are analyzed using dtypes function to see the data type of the continuous variables.
From the above image we can conclude that the continuous data consists of decimal numbers (float datatype).
Some of the categorical variables are analyzed using count plot to see the distribution of the data. Some of the plots obtained are as follows:
As the categorical variables cat116 and cat113 have many levels, it is unimportant and has no effect in predicting the target. Hence, these columns are dropped.
   
     The above plots show that these variables have biased levels(less entropy), hence these are considered unimportant, and these variables are dropped.
After dropping the unimportant variables, the categorical variables which resulted as important have to be converted to numeric. Categorical data has different levels for example, A, B and so on. If these categorical variables are converted to integers say 1 for A, 2 for B and so on, there are

chances that the model considers 1 superior to 2 because of the numeric precedence causing erroneous results. To avoid this, the categorical variables are converted to binary form using get dummies function of pandas. and hence the categorical variables are converted to numeric by using
Using the head function we see the first five records of the dataset to check if the categorical variables are converted or not.
 This shows that the categorical variables are converted to binary values.
To analyze the continuous variables, we use the describe function to get the seven numbers summary of the values, namely count, mean, standard deviation, minimum, first quartile, second quartile(median), third quartile and maximum. The result obtained for some of the continuous variables are as follows:
 As there is no validation dataset, we split the training dataset into train and validate dataset in the ratio of 75:25.

# PROPOSED SOLUTION AND METHODS
In order to predict the loss value, we used regression to predict the loss value. The regression algorithm used are as follows:
• Linear Regression
• Random Forest Regression
• Gradient Boosting Regressor

# Linear Regression:
The first model used is multi-linear regression. In this method, we predict the target value based on a number of predictor variables.
We use the Linear Regression library from pyspark.ml.regression. We create an instance of the model and fit it on the training dataset by setting the number of iterations to 1000. This model is evaluated by fitting on the validation dataset and metrics were analyzed as follows:
   
# Random Forest Regression
Random Forest model is an ensemble method of combinations of decision trees. In this method, a new dataset is not created but rather a group of decision trees work on different bootstrap samples of dataset with random splitting criteria for splitting the node. We used this model to predict the loss values using the parameter values. The results of each decision tree are averaged, and results are obtained as a combined value.

# Gradient Boosting Regressor
Gradient Boosting Regressor is a combination of gradient descent and Boosting. It is an ensemble method of weak models. These weak models are typically, decision trees. This model has three components, a loss functions, weak learner and an additive model to reduce loss by adding models.
    
# EXPERIMENTAL RESULTS AND ANALYSIS
The metrics we used are Mean Absolute Error, Root Mean Squared Error and R square. R square value is the ratio of sum of square of regression by the sum of square of total. As the R square value increases with the increase in features, it is not considered a useful metrics in evaluating a model. Hence, we use the Mean Absolute Error as the primary metric.
The loss value of the training dataset is analysed to get the summary of its values, the loss value varies from a minimum of 0.67 to maximum value of 121,012.25. Hence, the mean absolute error we obtained is in thousands.
The results obtained for each of the models used is as follows:
     
   For Gradient Boosting Regressor, we change the maximum number of iterations, Step Size in different combinations and the results are tabulated as follows:
The best loss value (Mean Absolute Error = 1225.3) was obtained when maximum iterations is 50 and step size is 0.3.
# FUTURE IMPROVEMENTS
A Co-Relation metric can be generated with each feature column and loss label column. This metric would be in range of -1.0 to 1.0. Any column whose co-relation metric less than a threshold for example 0.6 can be discarded this can further enhance the accuracy.
 
# CONCLUSION
We implemented three models on the dataset namely, Linear Regression, Random Forest Regression and Gradient Boosting Regression. From the results obtained above, we can conclude that Gradient Boosting Regressor gave the best results. As Mean Absolute Error is the competition’s official metric, it is our primary evaluation metric. The following table summarizes the results obtained.
      Regression Technique
     MAE
   Linear Regression
 1318.23
     Random Forest Regression
  1408.14
     Gradient Boosted Regressor
     1225.3
     
# TOOLS AND LANGUAGES USED:
• Pyspark is used as coding language.SparkML is used to import regression models.
• Scikit learn is used for principal component analysis.
• Pandas is used to read from and write into data frames, .csv files and to perform various
operations on the dataframes.
• Seaborn is used to plot the countplots.
  
# REFRENCES
https://www.allstate.com/about/general-information.aspx
https://seaborn.pydata.org /tutorial/categorical.html https://seaborn.pydata.org/generated/seaborn.countplot.html https://spark.apache.org/docs/0.9.0/python-programming-guide.html https://spark.apache.org/docs/2.2.0/ml-classification-regression.html https://www.kaggle.com/c/allstate-claims-severity http://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.feature.PCA https://spark.apache.org/docs/2.3.0/mllib-dimensionality-reduction.html
