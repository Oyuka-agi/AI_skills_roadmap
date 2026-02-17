#checking coursera data
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('/Users/era/Desktop/SPRING_2026/243_analytics_lab/AI_skills_roadmap/data/raw/skills/coursera_full_2026.csv')

# Check shapes of coursera dataset / inspect data
print(f"\nDataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

#column names
print('\nColumn names:', df.columns.tolist())
print('\nData types:', df.dtypes)

#inspect missing values
print('\nMissing values:', )
missing=df.isnull().sum()
missing_percent=(missing/len(df))*100

missing_table=pd.DataFrame({
    "Missing Count": missing, 
    "Missing %": missing_percent.round(2)
})

print(missing_table)

#inspect duplicates
duplicates=df.duplicated().sum()
print(f"\nNumber of duplicated rows: {duplicates}")

#numerical summary
print('\nNumerical Summary:', df.describe())

print('\nCategorical columns:')

categorical_cols=df.select_dtypes(exclude=[np.number]).columns
print("Categorical columns:", categorical_cols.tolist())

for col in categorical_cols:
    print(f"\nTop values for {col}:")
    print(df[col].value_counts().head(5))

print('\nUnique Values:')
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")



mostly_empty = missing_percent[missing_percent > 50]
if len(mostly_empty) > 0:
    print("\nColumns with >50% missing data:")
    print(mostly_empty)
else:
    print("\nNo columns have more than 50% missing data.")

print("\nEDA Completed Successfully!")


#visuals:
# ==============================
# VISUAL EDA SECTION
# ==============================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ------------------------------
# 1. MISSING VALUES BAR CHART
# ------------------------------
plt.figure(figsize=(10,5))
missing_percent.sort_values(ascending=False).plot(kind="bar")
plt.title("Missing Data Percentage by Column")
plt.ylabel("Percent Missing")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("missing_values.png")
plt.show()
print('graph worked')


# ------------------------------
# 2. TOP ORGANIZATIONS
# ------------------------------
top_orgs = df["Organization"].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_orgs.values, y=top_orgs.index)
plt.title("Top 10 Course Providers")
plt.xlabel("Number of Courses")
plt.ylabel("Organization")
plt.tight_layout()
plt.savefig("top_organizations.png")
plt.show()


# ------------------------------
# 3. LANGUAGE DISTRIBUTION
# ------------------------------
top_languages = df["Language"].value_counts().head(8)

plt.figure(figsize=(6,6))
plt.pie(top_languages.values, labels=top_languages.index, autopct='%1.1f%%')
plt.title("Course Language Distribution")
plt.tight_layout()
plt.savefig("language_distribution.png")
plt.show()


# ------------------------------
# 4. RATINGS DISTRIBUTION
# ------------------------------
ratings_clean = df["Ratings"].dropna()

plt.figure(figsize=(8,5))
sns.histplot(ratings_clean, bins=20, kde=True)
plt.title("Distribution of Course Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Courses")
plt.tight_layout()
plt.savefig("ratings_distribution.png")
plt.show()


# ------------------------------
# 5. PRICE BREAKDOWN
# ------------------------------
price_counts = df["Price"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(price_counts.values, labels=price_counts.index, autopct='%1.1f%%')
plt.title("Free vs Subscription Courses")
plt.tight_layout()
plt.savefig("price_breakdown.png")
plt.show()


print("\nVisualizations saved to project folder!")