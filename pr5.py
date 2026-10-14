import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor

# =========================
# Загрузка данных
# =========================

df = pd.read_csv("youtube_transactions.csv", header=None)

df.columns = ["time", "interests"]

# Создаем временной ряд для Gaming
df["Gaming"] = df["interests"].apply(
    lambda x: 1 if "Gaming" in x else 0
)

ts = df["Gaming"]

# =========================
# Визуализация ряда
# =========================

plt.figure(figsize=(12, 5))
plt.plot(ts)
plt.title("Интерес к Gaming")
plt.xlabel("Время")
plt.ylabel("Interest")
plt.show()

# =========================
# Проверка стационарности
# =========================

result = adfuller(ts)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Ряд стационарный")
else:
    print("Ряд нестационарный")

# =========================
# Декомпозиция
# =========================

decomposition = seasonal_decompose(
    ts,
    model='additive',
    period=20
)

decomposition.plot()
plt.show()

# =========================
# Подготовка данных
# =========================

# Создаем лаги
data = pd.DataFrame()
data["y"] = ts

for lag in range(1, 6):
    data[f"lag_{lag}"] = data["y"].shift(lag)

data = data.dropna()

X = data.drop("y", axis=1)
y = data["y"]

# Train/Test split
train_size = int(len(data) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# =========================
# ARIMA
# =========================

train_ts = ts[:train_size]
test_ts = ts[train_size:]

arima_model = ARIMA(train_ts, order=(1, 0, 1))
arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(
    steps=len(test_ts)
)

arima_mae = mean_absolute_error(
    test_ts,
    arima_forecast
)

print("\nARIMA MAE:", arima_mae)

# =========================
# Gradient Boosting
# =========================

gbr = GradientBoostingRegressor()

gbr.fit(X_train, y_train)

gbr_pred = gbr.predict(X_test)

gbr_mae = mean_absolute_error(
    y_test,
    gbr_pred
)

print("Gradient Boosting MAE:", gbr_mae)

# =========================
# LightGBM
# =========================

lgbm = LGBMRegressor()

lgbm.fit(X_train, y_train)

lgbm_pred = lgbm.predict(X_test)

lgbm_mae = mean_absolute_error(
    y_test,
    lgbm_pred
)

print("LightGBM MAE:", lgbm_mae)

# =========================
# Сравнение моделей
# =========================

models = ["ARIMA", "Gradient Boosting", "LightGBM"]
mae_scores = [arima_mae, gbr_mae, lgbm_mae]

plt.figure(figsize=(8, 5))

plt.bar(models, mae_scores)

plt.title("Сравнение моделей по MAE")
plt.ylabel("MAE")

plt.show()

# =========================
# График прогнозов
# =========================

plt.figure(figsize=(12, 5))

plt.plot(y_test.values, label="Real")

plt.plot(gbr_pred, label="Gradient Boosting")
plt.plot(lgbm_pred, label="LightGBM")

plt.legend()

plt.title("Сравнение прогнозов")

plt.show()