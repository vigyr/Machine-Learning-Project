
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

# CIS5560 Term Project Tutorial


------
#### Authors: [Vignesh Srinivas](https://www.linkedin.com/in/vigyr/); [Natya Srinivasan](https://www.linkedin.com/in/natya-srinivasan-04a47aa0/);  [Abhishek Shah](https://www.linkedin.com/in/abhishek-shah-bb3179a3/)

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/19/2017



# Creating Regression Models to Predict Score of a Restaurant

In this exercise, you will implement a regression model using *Linear Regression* and *Decision Tree Regression* to **predict the Score of a Resutaurant** - Restaurants in LA County. 

### **Pre-requisites:**

1. A Spark cluster, with default configuration as part of Databricks community edition.
2. Dataset for LA County Restaurant Violations. Available to download here : https://drive.google.com/file/d/0B-cqjuwpLeY4c1MxUy1JOGJlcEk/view?usp=sharing
3. Databricks community edition account. Signup for free here : https://databricks.com/try-databricks
4. SQL Source code which will be used for data preprocessing. Available to download here : https://drive.google.com/file/d/0B-cqjuwpLeY4OGx3dktpU0JiLVE/view?usp=sharing

### Creating a Cluster
Sign into your databricks account and go to Clusters option on the left and click on create cluster. Give the cluster name and click create cluster. The create cluster window will be like below, <br>
<img align="left" src="https://raw.githubusercontent.com/vigyr/Calstatela/master/Graphs/Cluster.JPG" style="width: 600px;" />
<br> <br>

These are the configuration options for the cluster, <br>
**Spark Version :** Spark 2.1 (Auto-updating, Scala 2.10) <br>
**Memory –** 6GB Memory , 0.88 Cores, 1 DBU <br>
**File System –** DBFS (Data Bricks File System)



***
### Overview
You should follow the steps below to build, train and test the model from the source data:

1. Import the Restaurant.csv as table in databricks.
2. Change the datatype for all the columns as required. 
3. Preprocess the data by removing outliers.
4. Prepare the data with the features (input columns, output column as label)
5. Split the data using data.randomSplit(): Training and Testing;Rename label to trueLabel in test.
6. Transform the columns to a vector using VectorAssembler
7. set features and label from the vector
8. Build a ***LinearRegression*** Model with the label and features
9. Build a Pipeline with 2 stages, VectorAssembler and LinearRegression. 
10. Use ParamGridBuilder() to build the parameter and TrainvalidationSplit() to evaluate the model. 
11. Train the model
12. Prepare the testing Data Frame with features and label from the vector.
13. Predict and test the testing Data Frame using the model trained at the step 11.
14. Compare the predicted result and trueLabel
15. Calculate RMSE for this model. 
16. Repeat the steps 8 - 15, but using ***Decision tree regression*** algorithm, instead of linear regression. 
17. Compare the RMSE for both models and predict the best model.

### Upload CSV as a Table 
The data for this exercise is provided as a CSV file containing details of Resutaurant violations. The data includes specific characteristics (or *features*) for each restaurant violations, as well as a *label* column indicating the final score of a Restaurant after the inspection. 

You will have to upload the csv file as a table in Databricks similar to below. 
<br>
<img align="left" src="https://raw.githubusercontent.com/vigyr/Calstatela/master/Graphs/Databricks-import.JPG" style="width: 600px;" />
<br>
***


Once the table is loaded, using preview table option, change the data type for each of the column as required. (INT for numeric columns and STRING for character columns) and save the table with a name given in this example as rest_vio.


### Import Spark SQL and Spark ML Libraries

First, import the libraries you will need:


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression , DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
```

### Load Source Data
Loading the data from table using a SQL syntax. <br> The source code is also available to download earlier in this tutorial. <br>
In our experiment we are considering the coulmns, ACTIVITY, NAME, TOTAL_VIOLATIONS, SCORE (label), VIOLATION_POINTS


```python
data = sqlContext.sql("SELECT ACTIVITY,NAME, CAST(count(VIOLATION_CODE) AS DOUBLE) as Total_violations,grade, CAST(score as DOUBLE) as label, CAST(sum(points) as DOUBLE) as Violation_points FROM rest_vio where score >= '65' group by NAME,ACTIVITY,grade,score")
```

### Removing Outliers
Considering the values for Total_violations that are less than 21. Filtering this using a where condition. <br>
For total_violations > 21, there are some inconsistencies in data, which will affect our overall prediction accuracy, so its better to elimate them for accurate predictions.


```python
dat = data.select("total_violations","label","ACTIVITY","NAME","grade","Violation_points").where((col("Total_violations") <= "21"))
```

### Prepare the Data
Most modeling begins with exhaustive exploration and preparation of the data. In this example, you will simply select Total_violations column as *features* as well as the **Score** column, which will be the *label* your model will predict.


```python
data_fea = dat.select("label","Total_violations")
```

### Split the Data
It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.


```python
splits = data_fea.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
print "We have %d training examples and %d test examples." % (train.count(), test.count())
```

### Build the Pipeline for Model 1
To train the regression model, you need to build a piepline that has two stages. First will be the Vector assembler that includes a vector of numeric features, and a label column. In this exercise, you will use the **VectorAssembler** class to transform the feature columns into a vector. Second stage will be regression algorithm you want to use. In this exercise, for first model you will use a ***Linear Regression algorithm*** - though you can use the same technique for any of the regression algorithms supported in the spark.ml API.


```python
vectorAssembler = VectorAssembler(inputCols=["Total_violations"], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
pipeline1 = Pipeline(stages=[vectorAssembler, lr])
```

### Tune Parameters to Train Model 1

You can tune parameters to find the best model for your data. A simple way to do this is to use TrainValidationSplit to evaluate each combination of parameters defined in a ParameterGrid against a subset of the training data in order to find the best performing parameters.

#### Regularization

is a way of avoiding Imbalances in the way that the data is trained against the training data so that the model ends up being over fit to the training data. In other words It works really well with the training data but it doesn't generalize well with other data. That we can use a regularization parameter to vary the way that the model balances that way.

#### Training ratio of 0.8

it's going to use 80% of the the data that it's got in its training set to train the model and then the remaining 20% is going to use to validate the trained model.

In ParamGridBuilder, all possible combinations are generated from regParam, maxIter, threshold. So it is going to try each combination of the parameters with 80% of the the data to train the model and 20% to to validate it.



```python
paramGrid1 = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
tvs1 = TrainValidationSplit(estimator=pipeline1, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid1, trainRatio=0.8)
model1 = tvs1.fit(train)
```

### Test the Model 1
Now you're ready to use the **transform** method of the model to generate some predictions. You can use this approach to predict score of a restaurant where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted score. 


```python
prediction1 = model1.transform(test)
# LinearRegression
predicted1 = prediction1.select("prediction", "trueLabel")
display(predicted1)
```

### Calculate the RMSE for Model 1
Using the evaluation metric as RMSE(Root Mean Squared Error), the Linear Regression model performance is calculated. 


```python
# LinearRegression: predictionCol="prediction", metricName="rmse"
evaluator1 = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse1 = evaluator1.evaluate(prediction1)
print "Root Mean Square Error (RMSE) For Linear Regression Model:", rmse1
```

### Result shows:
#### `Root Mean Square Error (RMSE): 1.6369`

The result is how on average - how much score - are in this spark prediction down to.

---
### Build the Pipeline for Model 2
Now, we will use ***Decision tree regression*** algorithm as our second model and build the pipeline for this model. The first stage will be the vector assembler we created earlier, and second stage will be the decision tree algorithm which can be created as below.


```python
vectorAssembler = VectorAssembler(inputCols=["Total_violations"], outputCol="features")
dt = DecisionTreeRegressor(labelCol="label", featuresCol="features")
pipeline2 = Pipeline(stages=[vectorAssembler, dt])
```

### Tune Parameters to Train Model 2
Similar to above, use the train validation split to evaluate each combination of parameters and then train the model. Similarly, with trainr atio of 0.8, it's going to use 80% of the the data that it's got in its training set to train the model and then the remaining 20% is going to use to validate the trained model.


```python
paramGrid2 = ParamGridBuilder().addGrid(dt.maxDepth, [2, 10]).build()
tvs2 = TrainValidationSplit(estimator=pipeline2, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, trainRatio=0.8)
model2 = tvs2.fit(train)
```

### Test the Model 2
Now you're ready to use the **transform** method of the model to generate some predictions. You can use this approach to predict score of a restaurant where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted score. 


```python
prediction2 = model2.transform(test)
predicted2 = prediction2.select("prediction", "trueLabel")
display(predicted2)
```

### Calculate the RMSE for Model 2
Using the evaluation metric as RMSE(Root Mean Squared Error), the Decision Tree Regression model performance is calculated. 


```python
evaluator2 = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse2 = evaluator2.evaluate(prediction2)
print "Root Mean Square Error (RMSE) For Decision Tree Regression Model %g" % rmse2
```

### Result shows:
#### `Root Mean Square Error (RMSE): 1.6096`

The result is how on average - how much score - are in this spark prediction down to.

## Comparing the Best Model
On comparing the RMSE values for both the models, Model 2 - Decision tree regression will be the accurate model with lower Root mean squared error value of *** 1.6096. Hence Decision tree regression model (Model 2) is accurate model for this experiment. *** 

---
### References

1. [Importing Tables in Databricks](https://docs.databricks.com/user-guide/tables.html)
2. [SQLContext functions](https://forums.databricks.com/topics/sqlcontext.html)
3. [Markdown Cells in Jupyter](http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Working%20With%20Markdown%20Cells.html)
4. [Markdown Cheatsheet](https://datascience.ibm.com/docs/content/analyze-data/markd-jupyter.html)
5. [Markdown Guide](https://help.ghost.org/hc/en-us/articles/224410728-Markdown-Guide)
