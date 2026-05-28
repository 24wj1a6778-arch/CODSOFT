import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('movies.csv')
print("Dataset loaded! Shape:", df.shape)
print(df.head(3))
print("\nColumn Names:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
df = df[['Name', 'Year', 'Duration', 'Genre', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
df = df.dropna(subset=['Rating'])
df['Duration'] = df['Duration'].astype(str).str.replace('min', '').str.strip()
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].astype(str).str.replace(',', '').str.strip()
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'] = df['Votes'].fillna(df['Votes'].median())
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'] = df['Year'].fillna(df['Year'].median())
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = df[col].fillna('Unknown')
df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0].strip())
le = LabelEncoder()
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = le.fit_transform(df[col].astype(str))

print("\nCleaned Dataset Shape:", df.shape)
print(df.head(3))
features = ['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
X = df[features]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2  = r2_score(y_test, lr_preds)
print(f"\nLinear Regression → MAE: {lr_mae:.2f} | R² Score: {lr_r2:.2f}")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2  = r2_score(y_test, rf_preds)
print(f"Random Forest      → MAE: {rf_mae:.2f} | R² Score: {rf_r2:.2f}")

# ── 15. Visualization 1: Actual vs Predicted (Random Forest) ──
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_preds, alpha=0.4, color='steelblue', edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings (Random Forest)")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
print("Chart saved: actual_vs_predicted.png")
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title("Feature Importance - What Affects Movie Ratings?")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Chart saved: feature_importance.png")
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=20, kde=True, color='coral')
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()
print("Chart saved: rating_distribution.png")
print("\n── Sample Movie Prediction ──")
sample = pd.DataFrame([[2015, 120, 3, 50000, 10, 5, 8, 2]], columns=features)
predicted_rating = rf.predict(sample)[0]
print(f"Predicted Rating for Sample Movie: {predicted_rating:.1f} / 10")