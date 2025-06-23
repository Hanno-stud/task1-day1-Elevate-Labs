# 🚢 Titanic Dataset - Data Cleaning & Preprocessing<br/>  

1 File, 2 
- [x]  Google Colab(.ipynb) & <br/>
- [x]  Python(.py) version

<br/>
<br/>
Welcome to this **comprehensive, step-by-step, hands-on project** on **data cleaning and preprocessing using the Titanic Dataset.**

This repository is part of an **AI & ML Internship task** and is designed to:
- Introduce beginners to real-world data preprocessing
- Guide readers intuitively from simple steps to deep technical practices
- Include visuals, code snippets, outputs, and plots in a human-readable and visually attractive way

---

## 📚 What Will You Learn from This Project?
- How to **clean raw data**
- How to **handle missing values** using statistics
- How to **encode categorical features** into numerical values
- How to **scale numerical features**
- How to **detect and remove outliers**
- Why preprocessing is essential for Machine Learning

---

# 🔗 Dataset Information
- **Dataset:** Titanic Dataset
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Manual Step:** Please download `Titanic-Dataset.csv` from the link and place it in the **same project folder**.

---

# 🛠️ Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 🚀 Step-by-Step Project Workflow

## 🔹 Step 1: Import Libraries and Load Dataset
We begin by loading the Titanic dataset and displaying its basic information.

### 🧩 Code Snippet
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')
```
<br/>
🖼️ Dataset Head
<br/>


## 🔹 Step 2: Explore Dataset Information
We explore the structure, datatypes, and null values.

### 🧩 Code Snippet
```python
print(df.info())
print(df.isnull().sum())
```
<br/>
🖼️ Dataset Info

🖼️ Null Values Detected
<br/>


## 🔹 Step 3: Display Mean and Median for Age and Fare
Before handling missing values, we calculate the mean and median.

### 🧩 Code Snippet
```python
age_mean = df['Age'].mean()
age_median = df['Age'].median()
fare_mean = df['Fare'].mean()
fare_median = df['Fare'].median()

print(f"Mean of Age: {age_mean}")
print(f"Median of Age: {age_median}")
```
<br/>
🖼️ Mean and Median Values
<br/>


## 🔹 Step 4: Handle Missing Values
- Fill missing Age with median.
- Fill missing Embarked with mode.
- Drop the Cabin column.

### 🧩 Code Snippet
```python
df.fillna({'Age': df['Age'].median()}, inplace=True)
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
```
<br/>
🖼️ Null Values After Cleaning
<br/>


## 🔹 Step 5: Encode Categorical Features
Label Encoding: Convert 'Sex' to numerical (male: 0, female: 1).

One-Hot Encoding: Convert 'Embarked' into dummy variables.

### 🧩 Code Snippet
```python
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```


## 🔹 Step 6: Feature Scaling
Use StandardScaler to normalize 'Age' and 'Fare'.

### 🧩 Code Snippet
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```
<br/>
🖼️ Scaled Dataset
<br/>


## 🔹 Step 7: Visualize Outliers
We plot boxplots for 'Age' and 'Fare' to detect outliers.

### 🧩 Code Snippet
```python
plt.figure(figsize=(10, 5))
sns.boxplot(y=df['Age'])
plt.title('Boxplot of Age')
plt.show()
```
🖼️ Boxplot of Age

### 🧩 Code Snippet
```python
plt.figure(figsize=(10, 5))
sns.boxplot(y=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()
```
<br/>
🖼️ Boxplot of Fare
<br/>


## 🔹 Step 8: Remove Outliers
We use the Interquartile Range (IQR) method to remove outliers in 'Fare'.

### 🧩 Code Snippet
```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Fare'] >= (Q1 - 1.5 * IQR)) & (df['Fare'] <= (Q3 + 1.5 * IQR))]
```
<br/>
🖼️ Final Dataset Shape
<br/>


## 🎯 Final Learning Points
- Proper data preprocessing improves model accuracy.

- Visualizing the data helps detect issues like missing values and outliers.

- Encoding and scaling are essential steps to prepare data for machine learning algorithms.


## ✅ Project Quality Checklist

This checklist ensures the project is robust, reproducible, user-friendly, and open for collaboration or future extension.

### 📁 Project Structure & Files
- [x] Jupyter Notebook (`.ipynb`) version for interactive exploration and documentation
- [x] Script version (`.py`) for production or automation use
- [x] Dataset (`Titanic-Dataset.csv`) included or well-documented
- [x] `README.md` with progressive explanation from beginner to advanced
- [x] Screenshot folder with visual examples of key outputs and plots

---

### 🧹 Data Preprocessing
- [x] Initial exploration of dataset shape, columns, and datatypes
- [x] Null value identification and logging
- [x] Mean and median of key columns displayed for transparency
- [x] Missing values handled appropriately (e.g., median for Age, mode for Embarked)
- [x] Redundant or heavily null columns (like Cabin) removed

---

### 🔠 Feature Engineering
- [x] Label Encoding for binary categorical variables (e.g., Sex)
- [x] One-Hot Encoding for multi-class categorical variables (e.g., Embarked)
- [x] Dataset rechecked after encoding for shape and column correctness

---

### 📏 Feature Scaling
- [x] Numerical features (Age, Fare) scaled using StandardScaler
- [x] Post-scaling dataset verified to have mean ≈ 0 and std ≈ 1

---

### 📊 Outlier Detection & Handling
- [x] Boxplots used for outlier visualization
- [x] Outliers removed using IQR method
- [x] Dataset shape compared before and after removal

---

### 📸 Visuals & Explainability
- [x] Screenshots included for:
  - Dataset head
  - Dataset info
  - Null values before/after
  - Mean/median calculation
  - Encoded dataset
  - Scaled dataset
  - Boxplots (Age & Fare)
  - Final dataset shape
- [x] Explanation accompanying each screenshot in README.md

---

### 📖 Documentation & Usability
- [x] Human-readable explanations included at each code step
- [x] Project explained in beginner-friendly tone, with increasing depth
- [x] Jargon-free sections for first-time learners
- [x] Clear file instructions and manual steps mentioned

---

### 🧪 Reproducibility & Execution
- [x] All steps executable independently without internet requirement (except dataset download)
- [x] Dataset filename and manual placement clearly mentioned
- [x] Code works consistently across notebook and script versions
- [x] No hardcoded paths or environment-specific variables

---

### 📦 Future-Proofing
- [x] File structure allows easy upgrade (e.g., model training pipeline can be added later)
- [x] Notebook clean, modular, and ready for extensions (e.g., EDA, ML models)
- [x] Encoding and scaling steps generic enough to work with similar datasets

---

### 💡 Educational Value
- [x] Answers to foundational ML interview questions included
- [x] Concepts explained with both theory and real code examples
- [x] Markdown cells used in notebook for clarity and readability

---

### 🧭 Suggested Next Steps (Optional)
- [ ] Add Exploratory Data Analysis (EDA) section with visual patterns
- [ ] Add ML model training (Logistic Regression or Random Forest)
- [ ] Evaluate model performance on preprocessed data
- [ ] Wrap the project as an installable Python package or CLI tool
- [ ] Dockerize or virtualize environment for deployment




<hr/>
🧾 <b>Note:</b>
To help different kinds of users, I've included two code formats:

1. **`titanic_preprocessing.py`** — A single Python script file containing the **entire workflow in one place**, perfect for automation or terminal-based execution.

2. **`task1_day1_elevate_labs.ipynb`** — A **Jupyter Notebook file** with all code blocks organized into **separate cells**, ideal for learning, experimenting, and visual output viewing.

> This gives the flexibility for both coders who prefer script files and learners who love the interactive notebook environment.
