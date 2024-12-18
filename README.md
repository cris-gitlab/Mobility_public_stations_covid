# Mobility_public_stations_covid
The main goal is to predict changes mobility in public stations area knowing the changes in other places from the same town during a part of covid period.

## Introduction
This is a project made for Learning Analytics class. The main objective is to use PySpark regression methods in order to predict changes in mobility in 
public stations ("public transport hubs"[1]) in Berlin, Germany. All data is taken from a BigQuery database form Google Colab.

For more information about the data: https://www.google.com/covid19/mobility/data_documentation.html [1]

## Used tehnologies
- python
- PySpark
- BigQuery from Google Cloud
- matplotlib
- Google Colab
- ChatGPT
  
## Workflow
We create a query to take all the relevant data for our case study and we run it.
```
query = """
SELECT date, retail_and_recreation_percent_change_from_baseline AS retail_and_recreation, grocery_and_pharmacy_percent_change_from_baseline AS grocery_and_pharmacy,
parks_percent_change_from_baseline AS parks, transit_stations_percent_change_from_baseline AS transit_stations,
workplaces_percent_change_from_baseline AS workplaces, residential_percent_change_from_baseline AS residential
FROM `bigquery-public-data.covid19_google_mobility_eu.mobility_report`
WHERE country_region = 'Germany'
AND sub_region_1 = 'Berlin'
"""

# Execute the query
query_job = client.query(query)

# Convert query result to a pandas DataFrame
results = query_job.to_dataframe()
```

### Preprocessing
Delete the entries that contain NULL/NaN values:
```
results = results.dropna()
```

Combine all the features in a single vector and sclae them using MinMaxScaler:
```
assembler2= VectorAssembler(inputCols=['retail_and_recreation', 'grocery_and_pharmacy', 'parks', 'workplaces', 'residential'],
                           outputCol='features')
data= assembler2.transform(data)

#use minMax scaler
from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scaleModel= scaler.fit(data)
prep_data= scaleModel.transform(data)
```
### Methods
- We will split the data in two subsets: train and test.
- We will use Linear regression, Decision tree regression, Gradient-boosted tree regression, Random Forest Regression.
- We will create a grid search for each method with different parameters.
- We will apply the previous defined grid search using a TrainValidationSplit object for every method. We fit the object using train data.
- We will evaluate each method using the test subset.

### Results
The table of results for each method with R2 and RMSE metrics:

|Method|R2|RMSE|
|---|---|---|
|Linear Regression|0.9137092832586994|4.389117300213218|
|Decision Tree Regression|0.925086966773904|4.089534162368974|
|Gradient-boosted Tree Regression|0.929996592978751|3.9532541740081615|
|Random Forest Regression|0.9491898099854995|3.3679881825070264|
