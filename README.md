# Cancer Detection Using KNN & Flask

## 📌 Overview
This project is a **Breast Cancer Detection System** built using the **K-Nearest Neighbors (KNN)** classifier. It helps physicians identify whether a tumor is **Benign** or **Malignant** based on 30 real-valued features extracted from cell nuclei images.

It includes:
- A **Flask web app** for user interaction
- A **PostgreSQL database** for data storage
- A fully trained and tuned **KNN model**

---

## 🚀 Project Workflow

### 1. Business & Data Understanding
- **Goal:** Support physicians in accurate cancer detection.
- **Success Criteria:**
  - Improve diagnosis accuracy to **96%**
  - Achieve **98%+ model accuracy**
  - Reduce costs and increase hospital revenue by **12%**
- **Dataset:** Breast cancer dataset with 569 patient records, 30 features, and 1 target (`diagnosis`).

---

### 2. Data Preprocessing
- **Missing Values:** Imputed using `SimpleImputer` with `mean` strategy.
- **Outliers:** Treated with `Winsorizer` from `feature_engine`.
- **Categorical Encoding:** `OneHotEncoder` via `DataFrameMapper` on `Sex` column.
- **Scaling:** Applied `MinMaxScaler`.

---

### 3. Machine Learning Model
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning:** Performed using `GridSearchCV` with odd `k` values from 3 to 49.
- **Model Selection Metric:** Accuracy Score
- **Best Model:** Saved using `pickle` as `knn.pkl`

---

### 4. PostgreSQL Database Integration
- **Database Name:** `wbcd_db`
- **Table:** Contains original patient data from CSV.
- **Connection:** Via `SQLAlchemy` and `psycopg2`
- **Schema:** Matches the original dataset.

---

### 5. Flask Web Application
- **Routes:**
  - `/`: Upload form for new data
  - `/predict`: Prediction results using saved KNN model
- **Functionality:**
  - Upload new patient data
  - Run predictions
  - Store predictions in database
- **Templates:**
  - `index.html` — for upload
  - `results.html` — for results

---

## 🛠 Tech Stack

| Area                  | Tools / Libraries                         |
|-----------------------|--------------------------------------------|
| Language              | Python                                     |
| Machine Learning      | scikit-learn, sklearn-pandas               |
| Web Framework         | Flask                                      |
| Database              | PostgreSQL, SQLAlchemy, psycopg2           |
| Preprocessing         | pandas, numpy, OneHotEncoder, MinMaxScaler |
| Deployment            | Pickle, Joblib                             |

---

## 🗂 Project Structure

# 📦 Project Structure: Cancer Detection KNN

- cancer-detection-knn/
  - app.py  
    📄 Flask web app for handling requests and predictions
  - model_training.py  
    📄 Script for preprocessing and training the KNN model
  - wbcd.csv  
    📊 Dataset file (Wisconsin Breast Cancer Data)
  - knn.pkl  
    🧠 Trained KNN model serialized with pickle
  - processed1  
    ⚙️ Preprocessing pipeline (step 1 – e.g., imputer, scaler)
  - processed2  
    ⚙️ Preprocessing pipeline (step 2 – e.g., feature selection)
  - requirements.txt  
    📦 List of dependencies required to run the project
  - README.md  
    📘 Project documentation (this file)
  - templates/  
    📁 Folder containing frontend HTML templates
    - index.html  
      📝 Web form to upload and input feature values
    - results.html  
      📊 Displays prediction results to the user



---

## 🧪 How to Run the Project

### Step 1: Clone the Repository
```
git clone https://github.com/your-username/cancer-detection-knn.git
cd cancer-detection-knn
```
### Step 2: Create Virtual Environment & Install Requirements
```python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```
### Step 3: Setup PostgreSQL
Create database wbcd_db

Update credentials in app.py and model_training.py:
conn_string = 'postgresql+psycopg2://postgres:your_password@localhost:5432/wbcd_db'

### Step 4: Train the Model & Load Data
python model_training.py

### Step 5: Run the Flask App
python app.py

## 📈 Results
Metric	            Value
Training Accuracy  	98.8%
Testing Accuracy	  97.4%
Best K	               21

Model selected using GridSearchCV

Evaluated with confusion matrix

High precision & recall

## 🧠 Future Scope
✅ Add PCA or dimensionality reduction
✅ Convert model into REST API
✅ Dockerize for deployment
✅ Add login/authentication system
✅ Deploy on cloud (AWS/GCP)

## 🙌 Acknowledgements
Thanks to the UCI Machine Learning Repository for the dataset and open-source libraries that made this project possible.
