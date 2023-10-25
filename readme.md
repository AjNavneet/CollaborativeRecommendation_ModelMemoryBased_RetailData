# Model and Memory-Based collaborative filtering Recommender System 

### Business Context

In our everyday lives, we often seek recommendations from friends who share similar tastes when it comes to products, movies, and more. We tend to trust recommendations from those with similar preferences. Collaborative filtering aims to replicate this concept by finding similarities between users and leveraging their preferences. Collaborative filtering is at the core of recommendation engines. It analyzes data about users with similar tastes and predicts the likelihood of an individual enjoying something.

---

### Aim

The primary objective of this project is to build a Recommender system using a variety of Model-Based and Memory-Based collaborative filtering techniques.

---

### Data Description
This dataset contains transactional records from a UK-based non-store online retail company. The company specializes in selling unique all-occasion gifts.

---

## Tech Stack
- Language: `Python`
- Libraries: `sklearn`, `surprise`, `pandas`, `matplotlib`, `scipy`, `numpy`, `pickle`

---

## Approach

1. **Data Description**: Understand the dataset.
2. **Data Cleaning**: Prepare the data for analysis.
3. **Memory-Based Approach**:
   - User-to-User Collaborative Recommendation
   - Item-to-Item Collaborative Recommendation
4. **Models**:
   - KNN model
   - Non-negative Matrix Factorization (NMF)
   - Co-Clustering model
5. **Evaluation Metrics**:
   - Root mean square error
   - Mean absolute error
   - Cross-validation score

---

## Modular Code Overview

1. `input`: Contains the data for analysis (e.g., `Rec_sys_data.xlsx`).
2. `src`: This is the heart of the project, containing modularized code for each step, organized as follows:
   - `ML_pipeline`: Functions organized in different Python files.
   - `engine.py`: Main script calling the functions from the `ML_pipeline` folder.
1. `output`: Contains the best-fitted model trained for the data, ready for future use.
2. `lib`: A reference folder containing the original IPython notebook used in the project.
3. `requirements.txt`: Lists all the required libraries with their respective versions. Install these libraries with `pip install -r requirements.txt`.

---

## Key Concepts Explored

1. Collaborative filtering and its types.
2. Memory-based approach.
3. User-to-User Collaborative Filtering and its implementation.
4. Item-to-Item Collaborative Filtering and its implementation.
5. Cosine similarity.
6. Model-based approach.
7. Performing a model-based approach using KNN.
8. Matrix Factorization.
9. Model-based approach using Matrix Factorization.
10. Detailed understanding of the Surprise library.
11. Different prediction algorithms in the Surprise library.
12. The Non-negative matrix factorization (NMF) model in the Surprise library.
13. Implementing Non-negative matrix factorization (NMF).
14. The Co-Clustering model in the Surprise library.
15. Implementing the Co-Clustering model.
16. Evaluating a recommendation system using the Surprise library.
17. CSR matrix and its significance.
18. Cross-validation in the context of recommendation systems.

---

