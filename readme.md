# Memory and Model Based collaborative filtering Recommender System 

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

