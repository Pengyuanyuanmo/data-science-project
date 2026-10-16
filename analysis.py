"""Reproduce the saved Ass1 housing analysis with local Python.
Adapted from PENG YUANYUAN's completed course notebook. Original methods retained.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent

DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv"
local_data = ROOT / "housing_course.csv"
df = pd.read_csv(local_data if local_data.exists() else DATA_URL)


# Original notebook cell 28
#Enter Your Code, Execute and take the Screenshot
df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)

df.describe()



# Original notebook cell 32
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mean())


# Original notebook cell 34
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].mean())


# Original notebook cell 40
plt.figure(figsize=(8, 6))
sns.boxplot(x='waterfront', y='price', data=df)
plt.title('House Price by Waterfront Status')
plt.xlabel('Waterfront (0 = No, 1 = Yes)')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.savefig(ROOT / "waterfront_prices.png", dpi=160)
plt.close()

print('Conclusion: houses without a waterfront view (0) have more price outliers in absolute count.')


# Original notebook cell 42
#Enter Your Code, Execute and take the Screenshot
plt.figure(figsize=(8, 6))
sns.regplot(x='sqft_above', y='price', data=df,
            scatter_kws={'alpha': 0.25}, line_kws={'color': 'red'})
plt.title('Price vs. Above-Ground Living Area')
plt.xlabel('Above-Ground Area (sq ft)')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.savefig(ROOT / "area_vs_price.png", dpi=160)
plt.close()

correlation = df['sqft_above'].corr(df['price'])
print(f'Pearson correlation: {correlation:.4f}')
print('Conclusion: sqft_above is positively correlated with price.')


# Original notebook cell 49
#Enter Your Code, Execute and take the Screenshot
X = df[['sqft_living']]
Y = df['price']

lm_sqft = LinearRegression()
lm_sqft.fit(X, Y)
r2_sqft = lm_sqft.score(X, Y)
print(f'R^2: {r2_sqft:.6f}')


# Original notebook cell 51
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# Original notebook cell 53
#Enter Your Code, Execute and take the Screenshot
X = df[features]
Y = df['price']

lm_multi = LinearRegression()
lm_multi.fit(X, Y)
r2_multi = lm_multi.score(X, Y)
print(f'R^2: {r2_multi:.6f}')


# Original notebook cell 55
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# Original notebook cell 57
#Enter Your Code, Execute and take the Screenshot
pipe = Pipeline(Input)
pipe.fit(df[features], df['price'])
r2_pipeline = pipe.score(df[features], df['price'])
print(f'Pipeline R^2: {r2_pipeline:.6f}')


# Original notebook cell 62
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# Original notebook cell 65
#Enter Your Code, Execute and take the Screenshot
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
r2_ridge = ridge_model.score(x_test, y_test)
print(f'Test R^2: {r2_ridge:.6f}')


# Original notebook cell 67
#Enter Your Code, Execute and take the Screenshot
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

ridge_poly = Ridge(alpha=0.1)
ridge_poly.fit(x_train_poly, y_train)
r2_ridge_poly = ridge_poly.score(x_test_poly, y_test)
print(f'Polynomial Ridge test R^2: {r2_ridge_poly:.6f}')


results = {"rows":len(df), "correlation_area_price":correlation,
 "in_sample_r2":{"sqft_living":r2_sqft,"multiple_linear":r2_multi,"polynomial_pipeline":r2_pipeline},
 "test_r2":{"ridge":r2_ridge,"polynomial_ridge":r2_ridge_poly},
 "note":"Original assignment methods retained: full-data mean imputation precedes splitting; this is a limitation."}
(ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
