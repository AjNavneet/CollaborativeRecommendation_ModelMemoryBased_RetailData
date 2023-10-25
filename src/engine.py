# Importing basic libraries
from surprise.prediction_algorithms import CoClustering, NMF
from ml_pipeline.preprocessing import create_csr_matrix, create_matrix, filter_data, drop_null, encode_units, similarity_score_dataframe
from ml_pipeline.utils import read_data, similar_user_recommendation, similar_items, similar_user_recommendation_knn, get_best_worst_predictions_by_model, find_cosine_similarity
from ml_pipeline.model import create_knn_model, create_and_evaluate_model, calculate_cross_validation
from ml_pipeline.preprocessing import data_prep_matrix_factorization, shortlisting_cust, splitting_data
import configparser

# Initialize the configuration parser
config = configparser.RawConfigParser()
config.read('..\\input\\config.ini')
DATA_DIR = config.get('DATA', 'data_dir')

# Reading data from the specified directory
data = read_data(DATA_DIR)

# Dropping rows with null values
data = drop_null(data)

# Filtering data to include rows with 'Quantity' greater than 0
data = filter_data(dataframe=data, column='Quantity', value=0)

# Memory-based approach

# User-to-user collaborative recommendation
print("\nUser-to-user collaborative recommendation:\n")

# Create a matrix of customer and product interactions
df = create_matrix(data, groupby_1='CustomerID', groupby_2='StockCode', aggr_measure='Quantity', set_index_by='CustomerID')

# Encode the purchase data
purchase = df.applymap(encode_units)

# Calculate cosine similarity between customers
user_similarity = find_cosine_similarity(purchase)

# Store the similarity scores in a dataframe
user_similarity_df = similarity_score_dataframe(user_similarity, purchase)

# Recommend items for a user with ID 12347
recommendation = similar_user_recommendation(user_id=12347, similarity_df=user_similarity_df, data=data, custid="CustomerID", productid="StockCode")

print('Items bought by Similar users based on Cosine Similarity using user-to-user collaborative filtering are : ', recommendation)

# Item-to-Item collaborative recommendation
print("\nItem-to-Item collaborative recommendation:\n")

# Create a matrix of customer and product interactions (transposed)
items_purchase = create_matrix(data, groupby_1='StockCode', groupby_2='CustomerID', aggr_measure='Quantity', set_index_by='StockCode')

# Encode the item purchase data
items_purchase = items_purchase.applymap(encode_units)

# Calculate cosine similarity between products
item_similarity = find_cosine_similarity(items_purchase)

# Store the similarity scores in a dataframe
item_similarity_df = similarity_score_dataframe(item_similarity, items_purchase)

# Recommend items similar to item with ID 22966
recommendation = similar_items(item_id=22966, similarity_df=item_similarity_df)

print('Similar items based on purchase behavior using item-to-item collaborative filtering => ', recommendation)

## KNN model
print("\nKNN model:\n")

# Create a CSR matrix of purchase data
purchase_matrix = create_csr_matrix(purchase)

# Create a KNN model
model = create_knn_model(purchase_matrix)

# Recommend items
output = similar_user_recommendation_knn(model=model, purchase=purchase, query_index=1497, data=data, custid="CustomerID", productid="StockCode")

print('Items bought by Similar users based on KNN:', output)

# Collaborative filtering using Matrix Factorization
print("\nCollaborative filtering using Matrix Factorization:\n")

# Data preparation for Matrix Factorization
df3 = data_prep_matrix_factorization(item_purchase=items_purchase)

# Shortlist customers & items based on the number of orders
df4 = shortlisting_cust(data=data, data_matrix_fact=df3, custid="CustomerID", productid="StockCode")

# Train-test-splitting
training_data, testing_data = splitting_data(df4)

# Implementing Non-negative Matrix Factorization
print("\nImplementing Non-negative Matrix Factorization:\n")

nmf_model, nmf_prediction, rmse, mae = create_and evaluate_model(model=NMF(), train=training_data, test=testing_data)

# Evaluation of Non-negative Matrix Factorization
print("Root mean square error of NMF : ", rmse)
print("Mean absolute error of NMF : ", mae)

# Cross-validation of NMF
print("Cross-validation results of NMF : ")
calculate_cross_validation(model=nmf_model, data=df4)

# Implementing Co-Clustering
print("\nImplementing Co-Clustering:\n")

co_clust_model, co_clust_prediction, rmse, mae = create_and_evaluate_model(model=CoClustering(), train=training_data, test=testing_data)

# Evaluation of Co-Clustering
print("Root mean square error of Co-Clustering : ", rmse)
print("Mean absolute error of Co-Clustering : ", mae)

# Cross-validation of Co-Clustering
print("Cross-validation results of Co-Clustering : ")
calculate_cross_validation(model=co_clust_model, data=df4)

# Best and worst prediction of the model
print("\nBest and worst prediction of the model:\n")

best_prediction, worst_prediction = get_best_worst_predictions_by_model(co_clust_prediction, training_data, ['item', 'customer', 'quantity', 'est', 'details'])

# Top 10 best predictions
print("\nTop 10 best predictions by Co-Clustering model => ")
print(best_prediction.head(10))

# Top 10 worst predictions
print("\nTop 10 worst predictions by Co-Clustering model => ")
print(worst_prediction.head(10))
