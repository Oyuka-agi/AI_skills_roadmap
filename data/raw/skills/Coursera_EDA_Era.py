### LOAD & INSPECT DATA
# checking coursera data
import numpy as np
import pandas as pd
import time

start_time=time.time()

# Load the data
df = pd.read_csv('/Users/era/Desktop/SPRING_2026/243_analytics_lab/AI_skills_roadmap/data/raw/skills/new_coursera_full_clean.csv')




# ### CONVERT UNIQUE VALUES INTO ENGLISH:   
# from deep_translator import GoogleTranslator

# translator = GoogleTranslator(source='auto', target='en')

# text_columns = df.select_dtypes(include=['object']).columns

# for col in text_columns:
#     print(f"Translating column: {col}")
    
#     unique_values = df[col].dropna().unique()
    
#     translation_map = {}
    
#     for val in unique_values:
#         try:
#             translation_map[val] = translator.translate(str(val))
#         except:
#             translation_map[val] = val
    
#     df[col] = df[col].map(translation_map)

# print("All categorical columns translated efficiently.")







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





### VISUAL EDA GRAPHS (nulls, organizations, languages, ratings)
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


# disregard # ------------------------------
# # 5. PRICE BREAKDOWN
# # ------------------------------
# price_counts = df["Price"].value_counts()

# plt.figure(figsize=(6,6))
# plt.pie(price_counts.values, labels=price_counts.index, autopct='%1.1f%%')
# plt.title("Free vs Subscription Courses")
# plt.tight_layout()
# plt.savefig("price_breakdown.png")
# plt.show()


# print("\nVisualizations saved to project folder!")


print('\ndetect languages\n:')

#detect languages
from langdetect import detect
import pandas as pd

def detect_lang(text):
    try:
        return detect(str(text))
    except:
        return "unknown"

df["lang_detected"] = df["Workload"].astype(str).apply(detect_lang)
print(df["lang_detected"].value_counts())


print('detecting languages done')

#converting units across languages
unit_map = {
    # hours
    "hours":"hour","hour":"hour","hrs":"hour","hr":"hour",
    "horas":"hour","heure":"hour","heures":"hour","stunden":"hour",
    "ora":"hour","ore":"hour","小时":"hour","시간":"hour","ساعة":"hour",

    # weeks
    "weeks":"week","week":"week","semanas":"week","semaines":"week",
    "wochen":"week","semaine":"week","週":"week","주":"week",

    # months
    "months":"month","meses":"month","mois":"month","monate":"month",
    "月":"month",

    # minutes
    "minutes":"minute","mins":"minute","min":"minute","minutos":"minute",
    "分钟":"minute"
}

#normalize text
import re

def normalize_units(text):
    if pd.isna(text):
        return text
    
    text = text.lower()
    
    for word, replacement in unit_map.items():
        text = re.sub(rf'\b{word}\b', replacement, text)
    
    return text

df["Workload_norm"] = df["Workload"].apply(normalize_units)

print(df["Workload_norm"].value_counts().head(40))

print('inspecting workload done')


#convert word to number
word_to_num = {
    "one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
    "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14"
}

def replace_word_numbers(text):
    if pd.isna(text): return text
    text = text.lower()
    for w,n in word_to_num.items():
        text = re.sub(rf'\b{w}\b', n, text)
    return text

df["Workload_norm"] = df["Workload_norm"].apply(replace_word_numbers)

#normalize connectors 
def normalize_ranges(text):
    if pd.isna(text): return text
    text = re.sub(r'\s*(to|a|–|—|~)\s*', '-', text)
    return text

df["Workload_norm"] = df["Workload_norm"].apply(normalize_ranges)

#normalize phrases
def normalize_weekly(text):
    if pd.isna(text): return text
    text = re.sub(r'(per week|a week|weekly|por semana)', 'hour/week', text)
    return text

df["Workload_norm"] = df["Workload_norm"].apply(normalize_weekly)

#remove filler words
def remove_noise(text):
    if pd.isna(text): return text
    noise_words = [
        "about","approximately","approx","of study","de curso",
        "divididas em","course","contenido","conteudo"
    ]
    for w in noise_words:
        text = text.replace(w,"")
    return re.sub(r'\s+',' ',text).strip()

df["Workload_norm"] = df["Workload_norm"].apply(remove_noise)

print(df["Workload_norm"].value_counts().head(50), 'workload normalization done')


### Kyras code- count missing nulls (test if i get same results)
missing_count = []
for col in df.columns:
    null_count = df[col].isnull().sum()
    zero_count = ((df[col] == 0) | (df[col] == '0')).sum() if df[col].dtype == 'object' or df[col].dtype == 'int64' or df[col].dtype == 'float64' else 0
    total_missing = null_count + zero_count
    missing_count.append({
        'Column': col,
        'Null_Count': null_count,
        'Zero_Count': zero_count,
        'Missing_Count': total_missing,
        'Missing_Percentage': round(total_missing / len(df) * 100, 2),
    })

missing_stats = pd.DataFrame(missing_count)
missing_stats = missing_stats.sort_values('Missing_Count', ascending=False)

print("\n" + "="*80)
print("Missing Values Statistics (Including 0 as Missing)")
print("="*80 + "\n")
print(missing_stats.to_string(index=False))
print('kyra code done, confirmed: missing counts match Kyra"s results')

### CORRELATION MATRIX
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()
print('correlation matrix done')

### OUTLIER DETECTION (using boxplot)
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Ratings"])
plt.title("Ratings Boxplot (Outlier Detection)")
plt.show()
print('boxplot done')

###More advanced analysis: comparisons
# rating vs organization
org_rating = df.groupby("Organization")["Ratings"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
org_rating.plot(kind="bar")
plt.title("Top Organizations by Average Rating")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()
print('org vs rating done')

# language vs rating
lang_rating = df.groupby("Language")["Ratings"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
lang_rating.plot(kind="bar")
plt.title("Ratings vs. Language")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show() 

### ZOOM into rating vs lang (rating 4.6-5)
plt.figure(figsize=(8,5))
lang_rating.plot(kind="bar")
plt.ylim(4.6,5)
plt.title("Ratings vs. Language")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show() 

print('language vs rating done')

#workload distribution after normalization
plt.figure(figsize=(8,5))
plt.ylabel("Frequency")
plt.xlabel("Normalized Workload")
df["Workload_norm"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Normalized Workload Categories")
plt.xticks(rotation=45)
plt.show()
print('workload distribution done')

###---------------------------------------------
# 1. Calculate both the mean and the count (sample size)
# Replace 'df' with your actual dataframe name
stats = df.groupby("Language")["Ratings"].agg(['mean', 'count']).sort_values('mean', ascending=False)

# 2. Setup the plot using your specific settings
plt.figure(figsize=(8,5))
# We plot the 'mean' column
ax = stats['mean'].plot(kind="bar", color="royalblue", alpha=0.8)

# 3. Add the sample size (count) text above each bar
for i, v in enumerate(stats['mean']):
    # Get the count for this specific language
    n_size = stats['count'].iloc[i]
    # Place the text: (x-position, y-position, string)
    ax.text(i, v + 0.005, f'n={int(n_size)}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Your specific formatting
plt.ylim(4.6, 4.925) # Your requested "Zoom"
plt.title("Ratings vs. Language")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
###---------------------------------------------


end_time=time.time()
print(f"\nTotal runtime: {round(end_time - start_time, 2)} seconds")

