# 1
import numpy as np
import pandas as pd

data_with_outliers = [10, 12, 14, 15, 16, 18, 100, 120, 130, 150]
print(data_with_outliers)
mean_before = np.mean(data_with_outliers)
median_before = np.median(data_with_outliers)
mode_before = pd.Series(data_with_outliers).mode()[0]

data_no_outliers = [x for x in data_with_outliers if x <= 50]
print(data_no_outliers)
mean_after = np.mean(data_no_outliers)
median_after = np.median(data_no_outliers)
mode_after = pd.Series(data_no_outliers).mode()[0]

print(f"Mean before outlier removal: {mean_before}")
print(f"Median before outlier removal: {median_before}")
print(f"Mode before outlier removal: {mode_before}")
print(f"Mean after outlier removal: {mean_after}")
print(f"Median after outlier removal: {median_after}")
print(f"Mode after outlier removal: {mode_after}")

# 2
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

test_scores = [78, 82, 90, 88, 74, 91, 92, 85, 80, 87, 95, 91, 79]
mean = stats.mean(test_scores)
median = stats.median(test_scores)
mode = stats.mode(test_scores)

plt.hist(test_scores, bins=6, color='skyblue', edgecolor='black')
plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean}')
plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median}')
plt.axvline(mode, color='b', linestyle='dashed', linewidth=1, label=f'Mode: {mode}')
plt.legend()
plt.title("Test Scores with Central Tendency Measures")
plt.xlabel("Test Score")
plt.ylabel("Frequency")
plt.show()

print(f"Mean: {mean}, Median: {median}, Mode: {mode}")

# 3
import numpy as np
import matplotlib.pyplot as plt

data = [12, 15, 16, 18, 19, 21, 23, 24, 30, 35]
range_value = np.ptp(data)
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
variance = np.var(data)
std_dev = np.std(data)

print(f"Range: {range_value}")
print(f"IQR: {iqr}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")

plt.boxplot(data, vert=False)
plt.title("Boxplot of Data")
plt.show()

# 4
from scipy import stats

before = [60, 65, 70, 72, 78]
after = [65, 70, 75, 80, 85]

t_stat, p_value = stats.ttest_rel(before, after)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
if p_value < 0.05:
    print("The study program was effective (reject the null hypothesis).")
else:
    print("The study program was not effective (fail to reject the null hypothesis).")

# 5
import numpy as np
from scipy import stats

def type_i_error():
    sample = np.random.normal(50, 10, 100)
    t_stat, p_value = stats.ttest_1samp(sample, 50)
    if p_value < 0.05:
        print("Type I Error: Rejecting a true null hypothesis")

def type_ii_error():
    sample = np.random.normal(60, 10, 100)
    t_stat, p_value = stats.ttest_1samp(sample, 50)
    if p_value >= 0.05:
        print("Type II Error: Failing to reject a false null hypothesis")

type_i_error()
type_ii_error()

# 6
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

years_of_experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
salary = np.array([40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000])

model = LinearRegression()
model.fit(years_of_experience, salary)

predicted_salary = model.predict([[5]])

plt.scatter(years_of_experience, salary, color='blue')
plt.plot(years_of_experience, model.predict(years_of_experience), color='red', linewidth=2)
plt.title("Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

print(f"Predicted salary for 5 years of experience: {predicted_salary[0]}")

# 7
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.uniform(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**2 + np.random.normal(0, 0.5, 100).reshape(-1, 1)

linear_model = LinearRegression()
linear_model.fit(X, y)

poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)  # Apply polynomial transformation
polynomial_model = LinearRegression()
polynomial_model.fit(X_poly, y)  # Fit polynomial regression model

y_linear_pred = linear_model.predict(X)
y_poly_pred = polynomial_model.predict(X_poly)  # Predict using transformed data

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_linear_pred, color='red', label='Linear Model (Underfitting)')
plt.plot(np.sort(X, axis=0), polynomial_model.predict(poly.transform(np.sort(X, axis=0))), color='green', label='Polynomial Model (Overfitting)')
plt.legend()
plt.title("Underfitting vs Overfitting")
plt.show()


# 8
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)
y_lasso_pred = lasso_model.predict(X)

linear_mse = mean_squared_error(y, y_linear_pred)
poly_mse = mean_squared_error(y, y_poly_pred)
lasso_mse = mean_squared_error(y, y_lasso_pred)

plt.scatter(X, y, color='blue')
plt.plot(X, y_linear_pred, color='red', label=f'Linear (MSE: {linear_mse:.2f})')
plt.plot(X, y_poly_pred, color='green', label=f'Polynomial (MSE: {poly_mse:.2f})')
plt.plot(X, y_lasso_pred, color='orange', label=f'Lasso (MSE: {lasso_mse:.2f})')
plt.legend()
plt.title("Comparison of Regression Models")
plt.show()

print(f"Linear Regression MSE: {linear_mse}")
print(f"Polynomial Regression MSE: {poly_mse}")
print(f"Lasso Regression MSE: {lasso_mse}")

# 9
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Dataset
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
marks_scored = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Calculate Pearson correlation
correlation = np.corrcoef(hours_studied.flatten(), marks_scored)[0, 1]

# Fit linear regression model
model = LinearRegression()
model.fit(hours_studied, marks_scored)

# Predictions
predictions = model.predict(hours_studied)

# Visualize
plt.scatter(hours_studied, marks_scored, color='blue')
plt.plot(hours_studied, predictions, color='red', linewidth=2)
plt.title("Hours Studied vs Marks Scored")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.show()

print(f"Pearson correlation coefficient: {correlation}")