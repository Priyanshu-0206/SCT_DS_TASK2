import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset (update path as needed)
df = pd.read_csv(r'C:\Users\Priyanshu\Downloads\titanic\train.csv')

# Basic info
print("Data Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# Checking missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())                        # Fill 'Age' missing values with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])        # Fill 'Embarked' missing values with mode
df = df.drop(columns=['Cabin'])                                        # Drop 'Cabin' due to excessive missing values

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# EDA Visualizations

# 1. Survival Count
sns.countplot(x='Survived', hue='Survived', data=df, palette='Set2', dodge=False, legend=False)
plt.title('Survival Count')
plt.show()

# 2. Survival by Sex
sns.countplot(x='Survived', hue='Sex', data=df, palette='pastel')
plt.title('Survival by Gender')
plt.show()

# 3. Survival by Passenger Class
sns.countplot(x='Survived', hue='Pclass', data=df, palette='coolwarm')
plt.title('Survival by Passenger Class')
plt.show()

# 4. Age Distribution (Histogram)
df['Age'].hist(bins=30, color='teal')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 5. Fare Distribution (Histogram)
df['Fare'].hist(bins=30, color='orange')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# 6. Survival by Embarked Location
sns.countplot(x='Embarked', hue='Survived', data=df, palette='Set1')
plt.title('Survival by Embarkation Port')
plt.show()

# 7. Correlation Heatmap (Fix Applied: Only Numeric Columns)
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Select only numeric columns
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
