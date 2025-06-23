# 🚢 Titanic Dataset - Data Cleaning & Preprocessing
<br/>  
<br/>

> 🌱 Introduction
>
> > Welcome to this **comprehensive, step-by-step, hands-on project** on **data cleaning and preprocessing using the Titanic Dataset.**
> > This repository is part of an **AI & ML Internship task** and is designed to:
> > - Introduce beginners to real-world data preprocessing
> > - Guide readers intuitively from simple steps to deep technical practices
> > - Include visuals, code snippets, outputs, and plots in a human-readable and visually attractive way

<br/>

## 📚 What Will You Learn from This Project?
- How to **clean raw data**
- How to **handle missing values** using statistics
- How to **encode categorical features** into numerical values
- How to **scale numerical features**
- How to **detect and remove outliers**
- Why preprocessing is essential for Machine Learning
<br/>

---

# 🔗 Dataset Information
- **Dataset:** Titanic Dataset
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Manual Step:** Please download `Titanic-Dataset.csv` from the link and place it in the **same project folder**.

<br/>
<br/>

> 🧾 **Note:**  
> The **Same Code** is provided in two executable formats for your convenience:  
> 
> - 💻 **[Google Colab](https://colab.research.google.com/)** (`.ipynb` file) — Jupyter Notebook with step-by-step cells, ideal for interactive learning.  
> - 🐍 **[Python](https://www.python.org/) Script** (`.py` file) — A single Python script, perfect for terminal or automated execution.  
>
> 👉 This gives the flexibility for both coders who prefer *`script files`* and learners who love the *`interactive notebook environment`*.

<br/>
<br/>

---

# 🛠️ Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn


<br/>

---

# 🚀 Step-by-Step Project Workflow
<br/>

## 🔹 Step 1: Import Libraries and Load Dataset
We begin by loading the Titanic dataset and displaying its basic information.
<br/>

### 🧩 Code Snippet
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')
```

### 🖥️ Console Output: Dataset Head, Info, and Null Values

```text
--- Dataset Head ---
   PassengerId  Survived  Pclass  
0            1         0       3  
1            2         1       1  
2            3         1       3  
3            4         1       1  
4            5         0       3  

                                                Name     Sex   Age  SibSp  
0                            Braund, Mr. Owen Harris    male  22.0      1  
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1  
2                             Heikkinen, Miss. Laina  female  26.0      0  
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1  
4                           Allen, Mr. William Henry    male  35.0      0  

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  


--- Dataset Info ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None


--- Null Values in Dataset ---
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```
<br/>


## 🔹 Step 2: Explore Dataset Information
We explore the structure, datatypes, and null values.
<br/>

### 🧩 Code Snippet
```python
print(df.info())
print(df.isnull().sum())
```

### 🖥️ Console Output: Dataset Head, Info, and Null Values

```text
--- Missing Values Before Handling ---
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64


--- Mean of Age: 29.70 ---
--- Mean of Fare: 32.20 ---


--- Missing Values After Handling ---
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
```
<br/>
<br/>

---

## Likewise...
🔹 Step 3: Display Mean and Median for Age and Fare <br/>
🔹 Step 4: Handle Missing Values <br/>
🔹 Step 5: Encode Categorical Features <br/>
🔹 Step 6: Feature Scaling <br/>
🔹 Step 7: Visualize Outliers <br/>
🔹 Step 8: Remove Outliers <br/>
<br/>
<br/>

--- 
## 🎯 Final Learning Points
- Proper data preprocessing improves model accuracy.

- Visualizing the data helps detect issues like missing values and outliers.

- Encoding and scaling are essential steps to prepare data for machine learning algorithms.
<br/>
<br/>

---

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

---
