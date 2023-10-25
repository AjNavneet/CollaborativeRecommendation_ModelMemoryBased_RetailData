from sklearn.neighbors import NearestNeighbors
from surprise import accuracy
from surprise.model_selection import cross_validate

# Function to create a KNN model
def create_knn_model(data):
    try:
        # Creating a KNN Model with the metric parameter set to euclidean distance
        model_knn = NearestNeighbors(metric='euclidean', algorithm='brute')

        # Fitting the model to the input data
        model_knn.fit(data)

    except Exception as e:
        print(e)
    else:
        return model_knn

# Function to create and evaluate a model
def create_and_evaluate_model(model, train, test):
    try:
        # Fit the provided model to the training data
        model = model.fit(train)

        # Generate predictions on the test data
        prediction = model.test(test)

        # Calculate RMSE (Root Mean Square Error) and MAE (Mean Absolute Error)
        rmse = accuracy.rmse(prediction)
        mae = accuracy.mae(prediction)
        
    except Exception as e:
        print(e)
    else:
        return model, prediction, rmse, mae

# Function to calculate cross-validation
def calculate_cross_validation(model, data):
    try:
        # Perform cross-validation on the provided model and data
        result = cross_validate(model, data, verbose=True)
    except Exception as e:
        print(e)
    else:
        return result
