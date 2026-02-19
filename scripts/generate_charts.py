# Auto-generated from titanic_analysis.ipynb
# Run: python scripts/generate_charts.py

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
os.chdir(BASE_DIR)


# --- cell 1 ---
try:
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("yasserh/titanic-dataset")
    dataset_csv = Path(path) / "Titanic-Dataset.csv"
    print("Path to dataset files:", path)
except ModuleNotFoundError:
    dataset_csv = BASE_DIR / "data" / "Titanic-Dataset.csv"
    if not dataset_csv.exists():
        raise RuntimeError(
            "kagglehub is not installed and no local dataset found at "
            f"{dataset_csv}. Install kagglehub or place Titanic-Dataset.csv in data/."
        )

# --- cell 2 ---
# Data manipulation
import pandas as pd
import numpy as np
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:
    LGBMClassifier = None

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create charts directory
os.makedirs('charts', exist_ok=True)

print("Libraries imported successfully!")

# --- cell 3 ---
# Load the dataset
df = pd.read_csv(dataset_csv)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
df.head()

# --- cell 4 ---
# Basic info
print("Dataset Info:")
print("="*50)
df.info()

# --- cell 5 ---
# Statistical summary
print("\nStatistical Summary:")
print("="*50)
df.describe()

# --- cell 6 ---
# Check for missing values
print("\nMissing Values:")
print("="*50)
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage': missing_percent
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df)

# --- cell 7 ---
# Visualize missing values
fig, ax = plt.subplots(figsize=(12, 6))
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    missing_data.plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Missing Values by Column', fontsize=16, fontweight='bold')
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Count of Missing Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('charts/01_missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No missing values found!")

# --- cell 8 ---
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# --- cell 9 ---
# Create a copy for cleaning
df_clean = df.copy()

print("Original shape:", df_clean.shape)

# Handle Age: Fill with median by Sex and Pclass
if 'Age' in df_clean.columns:
    df_clean['Age'] = df_clean.groupby(['Sex', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    print("Age missing values filled with median by Sex and Pclass")

# Handle Embarked: Fill with mode
if 'Embarked' in df_clean.columns:
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    print("Embarked missing values filled with mode")

# Handle Fare: Fill with median
if 'Fare' in df_clean.columns:
    df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
    print("Fare missing values filled with median")

# Handle Cabin: Create a binary feature for cabin availability
if 'Cabin' in df_clean.columns:
    df_clean['HasCabin'] = df_clean['Cabin'].notna().astype(int)
    print("Created HasCabin feature")

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# --- cell 10 ---
# Extract title from Name
if 'Name' in df_clean.columns:
    df_clean['Title'] = df_clean['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    df_clean['Title'] = df_clean['Title'].map(title_mapping)
    print("Title extracted and mapped")

# Family size
if 'SibSp' in df_clean.columns and 'Parch' in df_clean.columns:
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)
    print("FamilySize and IsAlone features created")

# Age groups
if 'Age' in df_clean.columns:
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=[0, 12, 18, 35, 60, 100],
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    print("AgeGroup feature created")

# Fare bins
if 'Fare' in df_clean.columns:
    df_clean['FareBin'] = pd.qcut(df_clean['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    print("FareBin feature created")

print("\nNew features:")
print(df_clean.columns.tolist())

# --- cell 11 ---
# Overall survival rate
if 'Survived' in df_clean.columns:
    survival_rate = df_clean['Survived'].value_counts(normalize=True) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    df_clean['Survived'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival Count', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Survived (0 = No, 1 = Yes)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['No', 'Yes'], rotation=0)
    
    # Pie chart
    axes[1].pie(df_clean['Survived'].value_counts(), labels=['Died', 'Survived'],
                autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], startangle=90)
    axes[1].set_title('Survival Rate', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/02_overall_survival.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSurvival Rate: {survival_rate[1]:.2f}%")
    print(f"Death Rate: {survival_rate[0]:.2f}%")

# --- cell 12 ---
# Survival by Sex
if 'Sex' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df_clean, x='Sex', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival by Gender', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Count')
    axes[0].legend(['Died', 'Survived'])
    
    # Survival rate
    survival_by_sex = df_clean.groupby('Sex')['Survived'].mean() * 100
    survival_by_sex.plot(kind='bar', ax=axes[1], color=['#3498db', '#e91e63'])
    axes[1].set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    for i, v in enumerate(survival_by_sex):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/03_survival_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSurvival rate by gender:")
    print(survival_by_sex)

# --- cell 13 ---
# Survival by Pclass
if 'Pclass' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df_clean, x='Pclass', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival by Passenger Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Passenger Class')
    axes[0].set_ylabel('Count')
    axes[0].legend(['Died', 'Survived'])
    
    # Survival rate
    survival_by_pclass = df_clean.groupby('Pclass')['Survived'].mean() * 100
    survival_by_pclass.plot(kind='bar', ax=axes[1], color=['#f39c12', '#9b59b6', '#1abc9c'])
    axes[1].set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Passenger Class')
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
    
    for i, v in enumerate(survival_by_pclass):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/04_survival_by_pclass.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSurvival rate by passenger class:")
    print(survival_by_pclass)

# --- cell 14 ---
# Age distribution
if 'Age' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    for survived in [0, 1]:
        axes[0].hist(df_clean[df_clean['Survived'] == survived]['Age'].dropna(),
                    alpha=0.6, bins=30, label=['Died', 'Survived'][survived])
    axes[0].set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Box plot
    sns.boxplot(data=df_clean, x='Survived', y='Age', ax=axes[1], palette=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Survived (0 = No, 1 = Yes)')
    axes[1].set_ylabel('Age')
    axes[1].set_xticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig('charts/05_survival_by_age.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAge statistics by survival:")
    print(df_clean.groupby('Survived')['Age'].describe())

# --- cell 15 ---
# Survival by Age Group
if 'AgeGroup' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df_clean, x='AgeGroup', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival by Age Group', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('Count')
    axes[0].legend(['Died', 'Survived'])
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    
    # Survival rate
    survival_by_age_group = df_clean.groupby('AgeGroup')['Survived'].mean() * 100
    survival_by_age_group.plot(kind='bar', ax=axes[1], color='skyblue')
    axes[1].set_title('Survival Rate by Age Group', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    for i, v in enumerate(survival_by_age_group):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/06_survival_by_age_group.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 16 ---
# Survival by Family Size
if 'FamilySize' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df_clean, x='FamilySize', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival by Family Size', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Family Size')
    axes[0].set_ylabel('Count')
    axes[0].legend(['Died', 'Survived'])
    
    # Survival rate
    survival_by_family = df_clean.groupby('FamilySize')['Survived'].mean() * 100
    survival_by_family.plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Family Size')
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    for i, v in enumerate(survival_by_family):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/07_survival_by_family_size.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 17 ---
# Survival by Embarked
if 'Embarked' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df_clean, x='Embarked', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival by Embarked Port', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Embarked Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
    axes[0].set_ylabel('Count')
    axes[0].legend(['Died', 'Survived'])
    
    # Survival rate
    survival_by_embarked = df_clean.groupby('Embarked')['Survived'].mean() * 100
    survival_by_embarked.plot(kind='bar', ax=axes[1], color=['#3498db', '#e91e63', '#f39c12'])
    axes[1].set_title('Survival Rate by Embarked Port', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Embarked Port')
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    for i, v in enumerate(survival_by_embarked):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/08_survival_by_embarked.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 18 ---
# Survival by Pclass and Sex
if 'Pclass' in df_clean.columns and 'Sex' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df_clean, x='Pclass', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Survival by Class and Gender', fontsize=14, fontweight='bold')
    
    # Facet grid
    survival_pclass_sex = df_clean.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
    survival_pclass_sex.plot(kind='bar', ax=axes[1], color=['#3498db', '#e91e63'])
    axes[1].set_title('Survival Rate by Class and Gender', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Passenger Class')
    axes[1].set_ylabel('Survival Rate')
    axes[1].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
    axes[1].legend(['Female', 'Male'])
    
    plt.tight_layout()
    plt.savefig('charts/09_survival_by_class_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 19 ---
# Heatmap: Survival by Age and Class
if 'Age' in df_clean.columns and 'Pclass' in df_clean.columns and 'Survived' in df_clean.columns:
    # Create age bins
    age_bins = pd.cut(df_clean['Age'], bins=[0, 12, 18, 35, 60, 100])
    
    # Create pivot table
    survival_age_class = pd.crosstab([age_bins, df_clean['Pclass']], df_clean['Survived'])
    survival_age_class['Survival Rate'] = survival_age_class[1] / (survival_age_class[0] + survival_age_class[1])
    
    pivot_data = survival_age_class['Survival Rate'].unstack(level=0)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'Survival Rate'})
    plt.title('Survival Rate Heatmap: Age vs Passenger Class', fontsize=14, fontweight='bold')
    plt.xlabel('Age Group')
    plt.ylabel('Passenger Class')
    plt.tight_layout()
    plt.savefig('charts/10_heatmap_age_class.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 20 ---
# Select numeric columns for correlation
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Create correlation matrix
correlation_matrix = df_clean[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numeric Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('charts/11_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature correlation with Survival
if 'Survived' in correlation_matrix.columns:
    survival_corr = correlation_matrix['Survived'].sort_values(ascending=False)
    print("\nCorrelation with Survival:")
    print(survival_corr)
    
    # Plot
    plt.figure(figsize=(10, 6))
    survival_corr[1:].plot(kind='barh', color='teal')
    plt.title('Feature Correlation with Survival', fontsize=14, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('charts/12_survival_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 21 ---
# Fare distribution
if 'Fare' in df_clean.columns and 'Survived' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    for survived in [0, 1]:
        axes[0].hist(df_clean[df_clean['Survived'] == survived]['Fare'],
                    alpha=0.6, bins=30, label=['Died', 'Survived'][survived])
    axes[0].set_title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fare')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].set_xlim([0, 300])
    
    # Box plot
    sns.boxplot(data=df_clean, x='Survived', y='Fare', ax=axes[1], palette=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Survived (0 = No, 1 = Yes)')
    axes[1].set_ylabel('Fare')
    axes[1].set_xticklabels(['No', 'Yes'])
    axes[1].set_ylim([0, 300])
    
    plt.tight_layout()
    plt.savefig('charts/13_survival_by_fare.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 22 ---
# Create a copy for modeling
df_model = df_clean.copy()

# Select features for modeling
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'HasCabin']

# Add Title if available
if 'Title' in df_model.columns:
    feature_cols.append('Title')

# Keep only necessary columns
available_features = [col for col in feature_cols if col in df_model.columns]
if 'Survived' in df_model.columns:
    df_model = df_model[available_features + ['Survived']]
else:
    df_model = df_model[available_features]

print(f"Features selected: {available_features}")
print(f"\nDataset shape: {df_model.shape}")
df_model.head()

# --- cell 23 ---
# Encode categorical variables
df_encoded = df_model.copy()

# Label encoding for binary categories
if 'Sex' in df_encoded.columns:
    df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked
if 'Embarked' in df_encoded.columns:
    embarked_dummies = pd.get_dummies(df_encoded['Embarked'], prefix='Embarked', drop_first=True)
    df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
    df_encoded.drop('Embarked', axis=1, inplace=True)

# One-hot encoding for Title
if 'Title' in df_encoded.columns:
    title_dummies = pd.get_dummies(df_encoded['Title'], prefix='Title', drop_first=True)
    df_encoded = pd.concat([df_encoded, title_dummies], axis=1)
    df_encoded.drop('Title', axis=1, inplace=True)

print("Encoded features:")
print(df_encoded.columns.tolist())
print(f"\nShape: {df_encoded.shape}")

# --- cell 24 ---
# Split features and target
if 'Survived' in df_encoded.columns:
    X = df_encoded.drop('Survived', axis=1)
    y = df_encoded['Survived']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTarget distribution:")
    print(y.value_counts(normalize=True))
else:
    print("Warning: 'Survived' column not found!")

# --- cell 25 ---
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts(normalize=True))
print(f"\nTest set class distribution:")
print(y_test.value_counts(normalize=True))

# --- cell 26 ---
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Data scaled successfully!")
print(f"\nScaled training data shape: {X_train_scaled.shape}")
print(f"Scaled test data shape: {X_test_scaled.shape}")

# --- cell 27 ---
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42),
}

if XGBClassifier is not None:
    models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
else:
    print("Warning: xgboost not installed; skipping XGBoost.")

if LGBMClassifier is not None:
    models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
else:
    print("Warning: lightgbm not installed; skipping LightGBM.")

print(f"Total models to train: {len(models)}")

# --- cell 28 ---
# Train and evaluate all models
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_mean,
        'ROC AUC': roc_auc
    })
    
    print(f"{name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, CV Mean: {cv_mean:.4f}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(results_df.to_string(index=False))

# --- cell 29 ---
# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
results_df.plot(x='Model', y='Accuracy', kind='barh', ax=axes[0, 0], color='skyblue', legend=False)
axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_ylabel('')

# F1-Score
results_df.plot(x='Model', y='F1-Score', kind='barh', ax=axes[0, 1], color='lightcoral', legend=False)
axes[0, 1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_ylabel('')

# Precision vs Recall
axes[1, 0].scatter(results_df['Recall'], results_df['Precision'], s=100, c=range(len(results_df)), cmap='viridis')
for idx, row in results_df.iterrows():
    axes[1, 0].annotate(row['Model'], (row['Recall'], row['Precision']), 
                       fontsize=8, ha='right', va='bottom')
axes[1, 0].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].grid(True, alpha=0.3)

# CV Mean
results_df.plot(x='Model', y='CV Mean', kind='barh', ax=axes[1, 1], color='lightgreen', legend=False)
axes[1, 1].set_title('Cross-Validation Mean Score', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('CV Mean Accuracy')
axes[1, 1].set_ylabel('')

plt.tight_layout()
plt.savefig('charts/14_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- cell 30 ---
# Select best model based on accuracy
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"Best Model: {best_model_name}")
print(f"\nBest Model Metrics:")
print(results_df.iloc[0].to_string())

# --- cell 31 ---
# Confusion Matrix for best model
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('charts/15_confusion_matrix_best.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Died', 'Survived']))

# --- cell 32 ---
# ROC Curve for best model
if hasattr(best_model, 'predict_proba'):
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_best)
    roc_auc = roc_auc_score(y_test, y_pred_proba_best)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/16_roc_curve_best.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 33 ---
# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['Importance'], color='teal')
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('charts/17_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
elif hasattr(best_model, 'coef_'):
    # For linear models
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': abs(best_model.coef_[0])
    }).sort_values('Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['Coefficient'], color='purple')
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
    plt.xlabel('Absolute Coefficient', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Feature Coefficients - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('charts/17_feature_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# --- cell 34 ---
# Get top 3 models
top_3_models = results_df.head(3)['Model'].tolist()
print(f"Top 3 models for hyperparameter tuning: {top_3_models}")

# --- cell 35 ---
# Define parameter grids for different models
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    },
    **({
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    } if XGBClassifier is not None else {}),
    **({
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 70],
            'subsample': [0.8, 0.9, 1.0]
        }
    } if LGBMClassifier is not None else {}),
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'linear', 'poly']
    },
    'Decision Tree': {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }
}

print("Parameter grids defined for hyperparameter tuning")

# --- cell 36 ---
# Perform RandomizedSearchCV for top 3 models
optimized_results = []

for model_name in top_3_models:
    if model_name in param_grids:
        print(f"\n{'='*80}")
        print(f"Optimizing {model_name}...")
        print(f"{'='*80}")
        
        # Get base model
        base_model = models[model_name]
        
        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grids[model_name],
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit the model
        random_search.fit(X_train_scaled, y_train)
        
        # Best parameters and score
        print(f"\nBest Parameters: {random_search.best_params_}")
        print(f"Best CV Score: {random_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred_opt = random_search.best_estimator_.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred_opt)
        test_f1 = f1_score(y_test, y_pred_opt)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Store results
        optimized_results.append({
            'Model': model_name,
            'Best Params': random_search.best_params_,
            'CV Score': random_search.best_score_,
            'Test Accuracy': test_accuracy,
            'Test F1-Score': test_f1,
            'Best Estimator': random_search.best_estimator_
        })
    else:
        print(f"\nNo parameter grid defined for {model_name}")

print("\n" + "="*80)
print("HYPERPARAMETER OPTIMIZATION COMPLETED")
print("="*80)

# --- cell 37 ---
# Compare optimized vs baseline models
comparison_data = []

for opt_result in optimized_results:
    model_name = opt_result['Model']
    baseline_acc = results_df[results_df['Model'] == model_name]['Accuracy'].values[0]
    optimized_acc = opt_result['Test Accuracy']
    improvement = ((optimized_acc - baseline_acc) / baseline_acc) * 100
    
    comparison_data.append({
        'Model': model_name,
        'Baseline Accuracy': baseline_acc,
        'Optimized Accuracy': optimized_acc,
        'Improvement (%)': improvement
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nBaseline vs Optimized Model Comparison:")
print(comparison_df.to_string(index=False))

# --- cell 38 ---
# Visualize improvement
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['Baseline Accuracy'], width, label='Baseline', color='lightblue')
bars2 = ax.bar(x + width/2, comparison_df['Optimized Accuracy'], width, label='Optimized', color='orange')

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Baseline vs Optimized Model Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('charts/18_baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
plt.show()

# --- cell 39 ---
# Select the best optimized model
best_optimized = max(optimized_results, key=lambda x: x['Test Accuracy'])

print("\n" + "="*80)
print("FINAL BEST MODEL")
print("="*80)
print(f"Model: {best_optimized['Model']}")
print(f"Test Accuracy: {best_optimized['Test Accuracy']:.4f}")
print(f"Test F1-Score: {best_optimized['Test F1-Score']:.4f}")
print(f"CV Score: {best_optimized['CV Score']:.4f}")
print(f"\nBest Parameters:")
for param, value in best_optimized['Best Params'].items():
    print(f"  {param}: {value}")

# Save the best model
final_model = best_optimized['Best Estimator']
final_model_name = best_optimized['Model']

# --- cell 40 ---
# Final confusion matrix
y_pred_final = final_model.predict(X_test_scaled)
cm_final = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title(f'Final Model Confusion Matrix - {final_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('charts/19_final_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Died', 'Survived']))

# --- cell 41 ---
# Final ROC curve
if hasattr(final_model, 'predict_proba'):
    y_pred_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]
    fpr_final, tpr_final, _ = roc_curve(y_test, y_pred_proba_final)
    roc_auc_final = roc_auc_score(y_test, y_pred_proba_final)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_final, tpr_final, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc_final:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Final Model ROC Curve - {final_model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/20_final_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- cell 42 ---
print("\n" + "="*80)
print("TITANIC DISASTER ANALYSIS - SUMMARY")
print("="*80)

print("\n1. DATA OVERVIEW:")
print(f"   - Total passengers: {len(df)}")
print(f"   - Survivors: {df['Survived'].sum()} ({df['Survived'].mean()*100:.2f}%)")
print(f"   - Deaths: {len(df) - df['Survived'].sum()} ({(1-df['Survived'].mean())*100:.2f}%)")

print("\n2. KEY INSIGHTS FROM EDA:")
print(f"   - Female survival rate: {df_clean[df_clean['Sex']=='female']['Survived'].mean()*100:.2f}%")
print(f"   - Male survival rate: {df_clean[df_clean['Sex']=='male']['Survived'].mean()*100:.2f}%")
print(f"   - 1st class survival rate: {df_clean[df_clean['Pclass']==1]['Survived'].mean()*100:.2f}%")
print(f"   - 3rd class survival rate: {df_clean[df_clean['Pclass']==3]['Survived'].mean()*100:.2f}%")

print("\n3. MODEL PERFORMANCE:")
print(f"   - Total models tested: {len(models)}")
print(f"   - Best baseline model: {best_model_name}")
print(f"   - Final optimized model: {final_model_name}")
print(f"   - Final test accuracy: {best_optimized['Test Accuracy']:.4f}")
print(f"   - Final test F1-Score: {best_optimized['Test F1-Score']:.4f}")

print("\n4. RECOMMENDATIONS:")
print("   - Gender and passenger class were the strongest predictors of survival")
print("   - Family size had a non-linear relationship with survival")
print("   - Age and fare also contributed to survival prediction")
print("   - The 'women and children first' policy was evident in the data")

print("\n5. FILES GENERATED:")
print(f"   - Total charts saved: {len([f for f in os.listdir('charts') if f.endswith('.png')])}")
print(f"   - Charts directory: charts/")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
