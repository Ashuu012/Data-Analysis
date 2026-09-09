# ============================================================
# Seasonal Agriculture Performance Analysis - VOIS Major Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style('whitegrid')

# 1. LOAD DATA
df = pd.read_csv("seasonal_agriculture_performance_dataset (1).csv")
print("Shape:", df.shape)
print(df.head())

# 2. EXPLORE
print(df.info())
print(df.isnull().sum()[df.isnull().sum() > 0])

# 3. CLEAN DATA (season-wise median imputation for missing values)
df_clean = df.copy()
for col in ['Rainfall_mm', 'Soil_Moisture_pct', 'Yield_Tonnes_Ha']:
    df_clean[col] = df_clean.groupby('Season')[col].transform(lambda x: x.fillna(x.median()))
print("Missing after cleaning:", df_clean[['Rainfall_mm','Soil_Moisture_pct','Yield_Tonnes_Ha']].isnull().sum().sum())

# 4. FEATURE ENGINEERING
df_clean['Profit_Margin_pct'] = df_clean['Profit_INR'] / df_clean['Revenue_INR'] * 100
df_clean['Loss_Flag'] = df_clean['Profit_INR'] < 0

# 5. SEASONAL COMPARISON
season_summary = df_clean.groupby('Season')[
    ['Yield_Tonnes_Ha', 'Profit_INR', 'Water_Used_m3', 'Disease_Pest_Risk_pct']
].mean().round(2)
print(season_summary)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=df_clean, x='Season', y='Yield_Tonnes_Ha', order=['Kharif','Rabi','Zaid'], ax=axes[0])
axes[0].set_title("Average Yield by Season")
sns.barplot(data=df_clean, x='Season', y='Profit_INR', order=['Kharif','Rabi','Zaid'], ax=axes[1])
axes[1].set_title("Average Profit by Season")
plt.tight_layout()
plt.show()

# 6. RELATIONSHIPS BETWEEN CONDITIONS AND OUTCOMES
corr = df_clean.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 7. CROP/REGION COMPARISON ACROSS SEASONS
pivot_profit = df_clean.pivot_table(values='Profit_INR', index='Crop', columns='Season', aggfunc='mean')
plt.figure(figsize=(7,5))
sns.heatmap(pivot_profit[['Kharif','Rabi','Zaid']], annot=True, fmt='.0f', cmap='RdYlGn', center=0)
plt.title("Avg Profit: Crop x Season")
plt.tight_layout()
plt.show()

# 8. STATISTICAL SIGNIFICANCE (ANOVA)
for metric in ['Yield_Tonnes_Ha', 'Profit_INR']:
    groups = [g[metric].dropna() for _, g in df_clean.groupby('Season')]
    f_stat, p_val = stats.f_oneway(*groups)
    sig = "significant" if p_val < 0.05 else "not significant"
    print(f"{metric}: F={f_stat:.2f}, p={p_val:.4f} -> {sig}")

# 9. KEY INSIGHTS
print("\nBest yield season:", season_summary['Yield_Tonnes_Ha'].idxmax())
print("Best profit season:", season_summary['Profit_INR'].idxmax())
print("Loss rate by season (%):\n", df_clean.groupby('Season')['Loss_Flag'].mean().mul(100).round(1))