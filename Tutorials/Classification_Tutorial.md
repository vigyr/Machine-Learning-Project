
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

# CIS5560 Term Project Tutorial


------
#### Authors: [Vignesh Srinivas](https://www.linkedin.com/in/vigyr/); [Natya Srinivasan](https://www.linkedin.com/in/natya-srinivasan-04a47aa0/);  [Abhishek Shah](https://www.linkedin.com/in/abhishek-shah-bb3179a3/)

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/19/2017



# Creating Classification Models to Predict Grade of a Restaurant

In this exercise, you will implement classifications models using *Decision tree Classifier* and *Random Forest classifier* to **predict the Grade of a Resutaurant** - Restaurants in LA County. 

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
5. Supply index values to the string type values using stringindexer().
6. Split the data using data.randomSplit(): Training and Testing;Rename label to trueLabel in test.
7. Transform the columns to a vector using VectorAssembler
8. set features and label from the vector
9. Build a ***Decision tree classifier*** Model with the label and features
10. Build a Pipeline with 2 stages, VectorAssembler and Decision tree classifier algorithm. 
11. Train the piepline model.
12. Prepare the testing Data Frame with features and label from the vector.
13. Predict and test the testing Data Frame using the model trained at the step 11.
14. Compare the predicted result and trueLabel
15. Calculate accuracy and test error for this model. 
16. Repeat the steps 9 - 15, but using ***Random forest classifier*** algorithm, instead of decision tree classifier. 
17. Compare the accuracy and test error for both models and predict the best model.

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

from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier , RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
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
Most modeling begins with exhaustive exploration and preparation of the data. In this example, you will select Total_violations and Violation_points columns as *features* as well as the **Grade** column, which will be the *label* your model will predict. <br>
As the Grade(label) column is of string datatype, we have to supply index values to the Grades in label column. This can be achieved by ***Stringindexer()*** function. The equivalent index values for grades are, **0:'A', 1:'B', 2:'C', 3:'SC'**


```python
indexer = StringIndexer(inputCol="grade", outputCol="Grade_Index")
indexed = indexer.fit(dat).transform(dat)
indexed_features = indexed.select("Total_violations",col("Grade_Index").alias("label"),"Violation_points")
ind_features = indexed_features.selectExpr("cast(Total_violations as int) Total_violations","cast(label as int) label","cast(Violation_points as int) Violation_points")
```

The metadata information of the string indexer can be obtained using the following command. 


```python
meta = [f.metadata for f in indexed.schema.fields if f.name == "Grade_Index"]
meta[0]
dict(enumerate(meta[0]["ml_attr"]["vals"]))
```

### Split the Data
It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.


```python
splits = ind_features.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
print "We have %d training examples and %d test examples." % (train.count(), test.count())
```

### Build the Pipeline for Model 1
To train the classification model, you need to build a piepline that has two stages. First will be the Vector assembler that includes a vector of numeric features, and a label column. In this exercise, you will use the **VectorAssembler** class to transform the feature columns into a vector. Second stage will be classification algorithm you want to use. In this exercise, for first model you will use a ***Decision Tree Classifier algorithm*** though you can use the same technique for any of the classification algorithms supported in the spark.ml API.


```python
vectorAssembler = VectorAssembler(inputCols=["Total_violations","Violation_points"], outputCol="features")
#Model1 - Decision Tree 
dt = DecisionTreeClassifier(labelCol="label", featuresCol= "features")
pipeline1 = Pipeline(stages=[vectorAssembler, dt])
```

### Train the Pipeline Model 1

Now you can train the pipeline model against your training data.


```python
model1 = pipeline1.fit(train)
```

### Test the Model 1
Now you're ready to use the **transform** method of the model to generate some predictions. You can use this approach to predict grade of a restaurant where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted grade. 


```python
prediction1 = model1.transform(test)
predicted1 = prediction1.select("prediction", "trueLabel")
display(predicted1)
```

### Calculate the Accuracy for Model 1
Using the evaluation metrics as Accuracy and Test error, the Decision tree classifier model performance is calculated. 


```python
evaluator1 = MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="accuracy")
accuracy1 = evaluator1.evaluate(prediction1)
print "Average Accuracy =", accuracy1
print "Test Error = ", (1 - accuracy1)

treeModel = model1.stages[1]
# summary only
print(treeModel)
```

### Result shows:
#### `The Accuracy for Decision tree classification model : 0.9974`
#### `The Test error for Decision tree classification model : 0.0025`

The result is how accurate - the grade of a restaurant - are in this spark prediction down to.

---
### Build the Pipeline for Model 2
Now, we will use ***Random forest classifier algorithm*** as our second model and build the pipeline for this model. The first stage will be the vector assembler we created earlier, and second stage will be the Random forest classifier algorithm which can be created as below.


```python
vectorAssembler = VectorAssembler(inputCols=["Total_violations","Violation_points"], outputCol="features")
rf = RandomForestClassifier(labelCol="label", featuresCol= "features", numTrees=30)
pipeline2 = Pipeline(stages=[vectorAssembler,rf])
```

### Train the Pipeline Model 2
Now you can train this second pipeline model against your training data.


```python
model2 = pipeline2.fit(train)
```

### Test the Model 2
Now you're ready to use the transform method of the model to generate some predictions. You can use this approach to predict grade of a restaurant where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted grade. 


```python
prediction2 = model2.transform(test)
predicted2 = prediction2.select("prediction", "trueLabel")
display(predicted2)
```

### Calculate the Accuracy for Model 2
Using the evaluation metrics as Accuracy and Test error, the Random Forest classifier model performance is calculated. 


```python
evaluator2 = MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="accuracy")
accuracy2 = evaluator2.evaluate(prediction2)
print "Average Accuracy =", accuracy2
print "Test Error = ", (1 - accuracy2)

rfModel = model2.stages[1]
print(rfModel)  # summary only
```

### Result shows:
#### `The Accuracy for Random Forest classification model : 0.9975`
#### `The Test error for Random Forest classification model : 0.0024`

The result is how accurate - the grade of a restaurant - are in this spark prediction down to.

## Comparing the Best Model
On comparing the accuracy values for both the models, Model 2 - Random forest classifier  is slightly better than Model1, because of higher accuracy rate.***(0.9975)***. ** Hence Random forest classifier model (Model 2) is accurate model for this experiment.** <br>
Other evaluation metrics like confusion matrix, recall, precision cannot be calculated in databricks for multi-class Classification. 

### References:
1. [String Indexer](http://stackoverflow.com/questions/33636944/preserve-index-string-correspondence-spark-string-indexer)
2. [Evaluation Metrics in Spark](https://spark.apache.org/docs/2.1.0/mllib-evaluation-metrics.html)
1. [Markdown Cells in Jupyter](http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Working%20With%20Markdown%20Cells.html)
1. [Markdown Cheatsheet](https://datascience.ibm.com/docs/content/analyze-data/markd-jupyter.html)
1. [Markdown Guide](https://help.ghost.org/hc/en-us/articles/224410728-Markdown-Guide)
