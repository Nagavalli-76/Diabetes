# 🩺 Diabetes Prediction

A machine learning project that predicts whether a patient is **likely to have diabetes** from basic medical measurements, using classification models built with scikit-learn (KNN, Decision Tree, and a Neural Network / MLP).

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key Insights](#-key-insights)
- [Models & Results](#-models--results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Limitations & Future Improvements](#-limitations--future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement

Early detection of diabetes helps patients get treatment sooner and reduces long-term health risks. This project uses patient health measurements to **classify whether a person has diabetes or not**.

The goal is to:
1. Explore which medical features (glucose, BMI, age, etc.) are linked to diabetes.
2. Build and compare classification models that predict the `Outcome` (1 = diabetic, 0 = non-diabetic).

---

## 📂 Dataset

- **File:** `diabetes.csv` (Pima Indians Diabetes dataset)
- **Size:** 768 rows (patients) × 9 columns, with no missing values (but many hidden zeros, see below)
- **Class balance:** about 35% diabetic (`Outcome = 1`) and 65% non-diabetic

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes risk score based on family history |
| `Age` | Age in years |
| **`Outcome`** | **Target: 1 = diabetic, 0 = not diabetic** |

> ⚠️ Columns such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contain `0` values that are not physically possible. These are really missing values and are handled during data preparation.

---

## 🔄 Project Workflow

### 1. Import & Inspect Data
- Loaded the dataset with Pandas and checked its shape, data types, and null counts.
- Reviewed summary statistics with `describe()`.

### 2. Data Visualization
- Correlation heatmap of all features.
- KDE (density) plots of `Pregnancies` and `Glucose` split by outcome.
- Violin plots of `Glucose` against `Outcome`.

### 3. Data Preparation
- Replaced the invalid `0` values with the **median** (`Glucose`, `BloodPressure`) or the **mean** (`BMI`, `SkinThickness`, `Insulin`).
- Separated the features (`x`) from the target (`y = Outcome`).
- Split the data into train and test sets (`test_size = 0.33`, `random_state = 42`).

### 4. Model Building
- **K-Nearest Neighbors (KNN):** tested `k` from 1 to 14, compared training vs. test accuracy, and chose `k = 9`.
- **Decision Tree:** an unrestricted tree and a depth-limited tree (`max_depth = 3`) to reduce overfitting.
- **MLP Neural Network:** trained on features scaled with `StandardScaler`.

### 5. Evaluation
- Compared training accuracy and test accuracy for each model.

---

## 💡 Key Insights

- **Glucose** separates the two classes most clearly: diabetic patients show noticeably higher glucose levels.
- Diabetic patients also tend to have a higher number of **pregnancies**, higher **BMI**, and are somewhat **older**.
- Several features contained **zeros that actually mean missing data**, so cleaning them is an important step before modelling.
- Limiting the depth of the Decision Tree is a simple way to reduce overfitting compared with an unrestricted tree.

---

## 📊 Models & Results

| Model | Settings |
|---|---|
| KNN | `n_neighbors = 9` |
| Decision Tree | `random_state = 0`, plus a version with `max_depth = 3` |
| MLP Neural Network | `max_iter = 1000`, `random_state = 42`, scaled inputs |

> ⚠️ **Important note on the reported accuracies:** in the current notebook, the model-training cell loads the **Iris dataset** (`load_iris()`) and overwrites `x_train`, `x_test`, `y_train`, `y_test`. As a result, the accuracy scores printed in the notebook come from **Iris, not from the diabetes data**, so they should not be reported as diabetes prediction results. See the first item under *Limitations & Future Improvements* for the fix, then re-run the notebook and add the real scores here.

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| KNN (k = 9) | _add after re-running_ | _add after re-running_ |
| Decision Tree | _add after re-running_ | _add after re-running_ |
| Decision Tree (max_depth = 3) | _add after re-running_ | _add after re-running_ |
| MLP | _add after re-running_ | _add after re-running_ |

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn (KNeighborsClassifier, DecisionTreeClassifier, MLPClassifier, StandardScaler, train_test_split)
- **Environment:** Jupyter Notebook

---

## 📁 Project Structure

```
Diabetes-Prediction/
│
├── Diabetes.ipynb      # Main analysis notebook
├── diabetes.csv        # Dataset
└── README.md           # Project documentation
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Diabetes-Prediction.git
   cd Diabetes-Prediction
   ```

2. **Install the dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open** `Diabetes.ipynb`, make sure `diabetes.csv` is in the same folder, and run all cells.

---

## 🚧 Limitations & Future Improvements

- **Fix the data leak between datasets:** remove the `load_iris()` block from the KNN cell and train on the diabetes `x_train`, `x_test`, `y_train`, `y_test` created earlier. Then re-run all models and update the results table.
- **Scale the features for KNN** (it is distance-based), using `StandardScaler` fit on the training data only.
- **Fix the scaling of the test set:** use `sc.transform(x_test)` instead of `sc.fit_transform(x_test)`, so the test data is scaled with the training statistics.
- **Impute after splitting:** compute the median/mean from the training set only, to avoid information from the test set leaking into the training data.
- **Use better metrics:** the classes are imbalanced (about 35% positive), so add a confusion matrix, precision, recall, F1-score, and ROC-AUC instead of relying on accuracy alone.
- **Use cross-validation and hyperparameter tuning** (`GridSearchCV`) instead of a single train/test split.
- Try other models such as Logistic Regression, Random Forest, or Gradient Boosting, and compare them with the baseline models.
- Handle class imbalance (class weights or SMOTE) if recall for diabetic patients is low.

---

## 👤 Author

**Kodidasu Veera Venkata Nagavalli**
B.Tech AI & ML, Aditya College of Engineering and Technology

- GitHub: [Nagavalli-76](https://github.com/Nagavalli-76)
- LinkedIn: [nagavalli-kodidasu-98384b340](https://www.linkedin.com/in/nagavalli-kodidasu-98384b340)
- Portfolio: [kvvn-portfolio.netlify.app](https://kvvn-portfolio.netlify.app)

---

⭐ If you found this project useful, consider giving it a star!
