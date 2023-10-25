import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

# Function to read an Excel file
def read_data(filepath):
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(e)
    else:
        return df

# Function to find k similar users based on cosine similarity
def similar_users(user_id, similarity_df, k=5):
    try:
        user = similarity_df[similarity_df.index == user_id]
        other_users = similarity_df[similarity_df.index != user_id]
        similarities = cosine_similarity(user, other_users)[0].tolist()
        indices = other_users.index.tolist()
        index_similarity = dict(zip(indices, similarities))
        index_similarity_sorted = sorted(index_similarity.items(), key=lambda x: x[1], reverse=True)
        top_users_similarities = index_similarity_sorted[:k]
        users = [u[0] for u in top_users_similarities]
    except Exception as e:
        print(e)
    else:
        return users

# Function to make recommendations using purchases of similar users
def similar_user_recommendation(user_id, similarity_df, data, custid, productid):
    try:
        similar_user = similar_users(user_id, similarity_df)
        simu_rec = []
        for j in similar_user:
            desc = data[data[custid] == j][productid].to_list()
            simu_rec.append(desc)
        flat_list = [item for sublist in simu_rec for item in sublist]
        final_list = list(dict.fromkeys(flat_list))
        ten_recs = random.sample(final_list, 10)
    except Exception as e:
        print(e)
    else:
        return ten_recs

# Function to find k similar items based on cosine similarity
def similar_items(item_id, similarity_df, k=5):
    try:
        item = similarity_df[similarity_df.index == item_id]
        other_items = similarity_df
        similarities = cosine_similarity(item, other_items)[0].tolist()
        indices = other_items.index.tolist()
        index_similarity = dict(zip(indices, similarities))
        index_similarity = list(index_similarity.keys())
        top_item_similarities = index_similarity[:k]
    except Exception as e:
        print(e)
    else:
        return top_item_similarities

# Function to find similar users using K-nearest neighbors (KNN)
def similar_users_knn(model, purchase, query_index):
    try:
        simu_knn = []
        distances, indices = model.kneighbors(purchase.iloc[query_index, :].values.reshape(1, -1), n_neighbors=5)
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for {0}:\n'.format(purchase.index[query_index]))
            else:
                print('{0}: {1}, with a distance of {2}:'.format(i, purchase.index[indices.flatten()[i]], distances.flatten()[i]))
                simu_knn.append(purchase.index[indices.flatten()[i]])
    except Exception as e:
        print(e)
    else:
        return simu_knn

# Function to make recommendations using purchases of similar users based on KNN
def similar_user_recommendation_knn(model, purchase, query_index, data, custid, productid):
    try:
        simu_knn = similar_users_knn(model, purchase, query_index)
        simu_rec = []
        for j in simu_knn:
            desc = data[data[custid] == j][productid].to_list()
            simu_rec.append(desc)
        flat_list = [item for sublist in simu_rec for item in sublist]
        final_list = list(dict fromkeys(flat_list))
        ten_recs = random.sample(final_list, 10)
    except Exception as e:
        print(e)
    else:
        return ten_recs

# Function to get the best and worst predictions of a model
def get_best_worst_predictions_by_model(prediction, trainset, col_list):
    def get_item_orders(uid):
        try:
            return len(trainset.ur[trainset.to_inner_uid(uid)])
        except ValueError:
            return 0

    def get_customer_orders(iid):
        try:
            return len(trainset.ir[trainset.to_inner_iid(iid)])
        except ValueError:
            return 0

    try:
        predictions_df = pd.DataFrame(prediction, columns=col_list)
        predictions_df['item_orders'] = predictions_df.item.apply(get_item_orders)
        predictions_df['customer_orders'] = predictions_df.customer.apply(get_customer_orders)
        predictions_df['err'] = abs(predictions_df.est - predictions_df.quantity)
        best_predictions = predictions_df.sort_values(by='err')[:10]
        worst_predictions = predictions_df.sort_values(by='err')[-10:]
    except Exception as e:
        print(e)
    else:
        return best_predictions, worst_predictions

# Function to find cosine similarity of a matrix
def find_cosine_similarity(dataframe):
    try:
        similarity = cosine_similarity(dataframe)
    except Exception as e:
        print(e)
    else:
        return similarity
