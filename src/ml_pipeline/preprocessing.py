import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split

# Function to clean data if it contains any null value and negative quantities.
def drop_null(dataframe):
    try:
        df1 = dataframe.dropna()
        df1 = df1[df1.Quantity > 0]
    except Exception as e:
        print(e)
    else:
        return df1

# Function to filter data based on a specified column and value.
def filter_data(dataframe, column, value):
    try:
        df1 = dataframe[dataframe[column] > value]
    except Exception as e:
        print(e)
    else:
        return df1

# Function to rename a column in a DataFrame.
def col_rename(dataframe, oldname, newname):
    try:
        dataframe.rename(columns={oldname: newname}, inplace=True)
    except Exception as e:
        print(e)
    else:
        return dataframe

# Function to encode quantity into binary values (0 or 1).
def encode_units(x):
    try:
        if x < 1:
            return 0  # Not purchased
        if x >= 1:
            return 1  # Purchased
    except Exception as e:
        print(e)

# Function to create a matrix from a DataFrame.
def create_matrix(dataframe, groupby_1, groupby_2, aggr_measure, set_index_by):
    try:
        user_purchase = (dataframe.groupby([groupby_1, groupby_2])[aggr_measure].sum().unstack().reset_index().fillna(0).set_index(set_index_by))
    except Exception as e:
        print(e)
    else:
        return user_purchase

# Function to merge two DataFrames on a specified column.
def merge_dataframes(dataframe1, dataframe2, on_col):
    try:
        dataframe = pd.merge(dataframe1, dataframe2, on=on_col, how="inner")
    except Exception as e:
        print(e)
    else:
        return dataframe

# Function to create a DataFrame of similarity scores.
def similarity_score_dataframe(similarity_df, dataframe):
    try:
        similarity_df = pd.DataFrame(similarity_df, index=dataframe.index, columns=dataframe.index)
    except Exception as e:
        print(e)
    else:
        return similarity_df

# Function to create a CSR matrix from a DataFrame.
def create_csr_matrix(dataframe):
    try:
        purchase_matrix = csr_matrix(dataframe.values)
    except Exception as e:
        print(e)
    else:
        return purchase_matrix

# Function to prepare data for matrix factorization.
def data_prep_matrix_factorization(item_purchase):
    try:
        df = item_purchase.stack().to_frame()
        df.reset_index(inplace=True)
    except Exception as e:
        print(e)
    else:
        return df

# Function to shortlist customers based on the number of orders.
def shortlisting_cust(data, data_matrix_fact, custid, productid):
    try:
        # Store all customer IDs in customers
        customers = data[custid]

        # Store all item descriptions in items
        items = data[productid]

        # Count the number of orders made by each customer
        count1 = Counter(customers)

        # Store the count and customer ID in a DataFrame
        countdf1 = pd.DataFrame.from_dict(count1, orient='index').reset_index()

        # Drop all customer IDs with fewer than 120 orders
        countdf1 = countdf1[countdf1[0] > 120]

        # Rename the index column as CustomerID for inner join
        countdf1.rename(columns={'index': custid}, inplace=True)

        # Count the number of times an item was ordered
        count2 = Counter(items)

        # Store the count and item description in a DataFrame
        countdf2 = pd.DataFrame from_dict(count2, orient='index').reset_index()

        # Drop all items that were ordered less than 120 times
        countdf2 = countdf2[countdf2[0] > 120]

        # Rename the index column as Description for inner join
        countdf2 = col_rename(countdf2, 'index', productid)

        # Apply inner join
        df = merge_dataframes(data_matrix_fact, countdf2, productid)
        df = merge_dataframes(df, countdf1, custid)

        # Drop columns which are not necessary
        df.drop(['0_y', '0_x'], axis=1, inplace=True)

        # Read the data in a format supported by the Surprise library.
        reader = Reader(rating_scale=(0, 5946))
        # The range has been set as 0, 5946 as the maximum value of quantity is 5946.

        # Load the Dataset in a format supported by the Surprise library.
        df = Dataset.load_from_df(df, reader)
    except Exception as e:
        print(e)
    else:
        return df

# Function to split data into train and test sets.
def splitting_data(data):
    try:
        # Perform train-test split on the dataset
        trainset, testset = train_test_split(data, test_size=0.2)
    except Exception as e:
        print(e)
    else:
        return trainset, testset
