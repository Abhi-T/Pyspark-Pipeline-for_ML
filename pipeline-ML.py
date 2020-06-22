from pyspark.sql import SparkSession
import pyspark.sql.types as tp

#read csv
spark=SparkSession.builder.master("local").appName("Pipeline-ML").config("spark.executor.memory","1g").\
    config("spark.core.max","2").getOrCreate()
# my_data=spark.read.csv("ind-ban-comment.csv",header=True)
# print(my_date.head())
# print(my_data.printSchema())

#creating custom schema
my_schema = tp.StructType([tp.StructField(name= 'Batsman',      dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Batsman_Name', dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Bowler',       dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Bowler_Name',  dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Commentary',   dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Detail',       dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Dismissed',    dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Id',           dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Isball',       dataType= tp.BooleanType(),   nullable= True),
    tp.StructField(name= 'Isboundary',   dataType= tp.BinaryType(),   nullable= True),
    tp.StructField(name= 'Iswicket',     dataType= tp.BinaryType(),   nullable= True),
    tp.StructField(name= 'Over',         dataType= tp.DoubleType(),    nullable= True),
    tp.StructField(name= 'Runs',         dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Timestamp',    dataType= tp.TimestampType(), nullable= True)
])

#read the data again with the defined schema
my_data=spark.read.csv("ind-ban-comment.csv",schema=my_schema,header=True)
print(my_data.printSchema())

#drop columns which are not required
my_data=my_data.drop(*['Batsman','Bowler','Id'])
print(my_data.columns)

#checking the dimention of the data
print((my_data.count(), len(my_data.columns)))

#get the summary of the numerical columns
print(my_data.select('Isball', 'Isboundary', 'Runs').describe().show())

# import pyspark.sql.functions as f
# data_agg = my_data.agg(*[f.count(f.when(f.isnull(c), c)).alias(c) for c in my_data.columns])
# print(data_agg.collect())

# print(my_data.select('Batsman_Name').distinct().count())

#Encode Categorical Variables using PySpark
from pyspark.ml.feature import StringIndexer, OneHotEncoder
SI_batsman=StringIndexer(inputCol='Batsman_Name',outputCol='Batsman_Index')
transform_data=SI_batsman.fit(my_data).transform(my_data)

# print(transform_data.select('Batsman_Index', 'Batsman_Name').groupBy('Batsman_Index', 'Batsman_Name').count().sort(
    # 'Batsman_Index').show())

#one hot encoding

SI_bowler=StringIndexer(inputCol='Bowler_Name', outputCol='Bowler_Index')
transform_data = SI_bowler.fit(transform_data).transform(transform_data)

# print(transform_data.select('Batsman_Name', 'Batsman_Index', 'Bowler_Name', 'Bowler_Index').show())

# create object and specify input and output column
OHE = OneHotEncoder(inputCol=['Batsman_Index', 'Bowler_Index'],outputCol=['Batsman_OHE', 'Bowler_OHE'])

# transform the data
OHE_data = OHE.fit(transform_data).transform(transform_data)

#view and transform the data
# OHE_data.select('Batsman_Name','Batsman_Index','Batsman_OHE').groupBy('Batsman_Name','Batsman_index','Batsman_OHE').count().sort('Bastman_Index').show()

#A vector assembler combines a given list of columns into a single vector column.

from pyspark.ml.feature import VectorAssembler
# specify the input and output columns of the vector assembler
assembler = VectorAssembler(inputCols=['Isboundary',
                                       'Iswicket',
                                       'Over',
                                       'Runs',
                                       'Batsman_Index',
                                       'Bowler_Index',
                                       'Batsman_OHE',
                                       'Bowler_OHE'],
                           outputCol='vector')

fill_null_data = OHE_data.fillna(0)
final_data = assembler.transform(fill_null_data)
print(final_data.show())

# A pipeline allows us to maintain the data flow of all the relevant transformations that are required to reach the end result.
# create a sample dataframe

sample_df = spark.createDataFrame([
    (1, 'L101', 'R'),
    (2, 'L201', 'C'),
    (3, 'D111', 'R'),
    (4, 'F210', 'R'),
    (5, 'D110', 'C')], ['id', 'category_1', 'category_2'])

print(sample_df.show())

from pyspark.ml import Pipeline

# define stage 1 : transform the column category_1 to numeric
stage_1 = StringIndexer(inputCol= 'category_1', outputCol= 'category_1_index')
# define stage 2 : transform the column category_2 to numeric
stage_2 = StringIndexer(inputCol= 'category_2', outputCol= 'category_2_index')
# define stage 3 : one hot encode the numeric category_2 column
stage_3 = OneHotEncoder(inputCols=['category_2_index'], outputCols=['category_2_OHE'])

# setup the pipeline
pipeline = Pipeline(stages=[stage_1, stage_2, stage_3])

# fit the pipeline model and transform the data as defined
pipeline_model = pipeline.fit(sample_df)
sample_df_updated = pipeline_model.transform(sample_df)

# view the transformed data
# print(sample_df_updated.show())

from pyspark.ml.classification import LogisticRegression

# create a sample dataframe with 4 features and 1 label column
sample_data_train = spark.createDataFrame([
    (2.0, 'A', 'S10', 40, 1.0),
    (1.0, 'X', 'E10', 25, 1.0),
    (4.0, 'X', 'S20', 10, 0.0),
    (3.0, 'Z', 'S10', 20, 0.0),
    (4.0, 'A', 'E10', 30, 1.0),
    (2.0, 'Z', 'S10', 40, 0.0),
    (5.0, 'X', 'D10', 10, 1.0),
], ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'label'])

# view the data
# print(sample_data_train.show())

# define stage 1: transform the column feature_2 to numeric
stage_1 = StringIndexer(inputCol= 'feature_2', outputCol= 'feature_2_index')

# define stage 2: transform the column feature_3 to numeric
stage_2 = StringIndexer(inputCol= 'feature_3', outputCol= 'feature_3_index')

# define stage 3: one hot encode the numeric versions of feature 2 and 3 generated from stage 1 and stage 2
stage_3 = OneHotEncoder(inputCols=[stage_1.getOutputCol(), stage_2.getOutputCol()],
                                 outputCols= ['feature_2_encoded', 'feature_3_encoded'])

# define stage 4: create a vector of all the features required to train the logistic regression model
stage_4 = VectorAssembler(inputCols=['feature_1', 'feature_2_encoded', 'feature_3_encoded', 'feature_4'],
                          outputCol='features')

# define stage 5: logistic regression model
stage_5 = LogisticRegression(featuresCol='features',labelCol='label')

# setup the pipeline
regression_pipeline = Pipeline(stages=[stage_1, stage_2, stage_3, stage_4, stage_5])

# fit the pipeline for the trainind data
model = regression_pipeline.fit(sample_data_train)
# transform the data
sample_data_train = model.transform(sample_data_train)

# view some of the columns generated
sample_data_train.select('features', 'label', 'rawPrediction', 'probability', 'prediction').show()
