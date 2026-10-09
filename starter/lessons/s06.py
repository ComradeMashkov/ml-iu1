"""Стоимость поездки: модель и прогноз. Run from starter or the course root. All data are synthetic."""

# S06-C01
import csv
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

data_dir = Path("../data")
if not data_dir.exists():
    data_dir = Path("../../data")
if not data_dir.exists():
    data_dir = Path("data")
if not data_dir.exists():
    data_dir = Path("starter/data")
print("Данные:", data_dir)


# S06-C02
with (data_dir / "taxi-trips-synthetic.csv").open() as file:
    rows = list(csv.DictReader(file))
train_rows = []
test_rows = []
for row in rows:
    if row["split"] == "train":
        train_rows.append(row)
    else:
        test_rows.append(row)
train_x = []
train_peak = []
train_y = []
for row in train_rows:
    train_x.append(float(row["distance_km"]))
    train_peak.append(int(row["peak"]))
    train_y.append(float(row["cost_rub"]))
print("Поездки train и test:", len(train_rows), len(test_rows))


# S06-C03
plt.scatter(train_x, train_y, c=train_peak, cmap="coolwarm")
plt.xlabel("Плановое расстояние, км")
plt.ylabel("Оплаченная цена, руб")
plt.title("48 учебных поездок: красный цвет означает час пик")
plt.grid(alpha=0.2)
plt.show()


# S06-C04
from sklearn.linear_model import LinearRegression

distance_features = []
for distance in train_x:
    distance_features.append([distance])
line_model = LinearRegression()
line_model.fit(distance_features, train_y)
mean_train_cost = sum(train_y) / len(train_y)
print("Линия: w и b:", line_model.coef_[0], line_model.intercept_)
print("Среднее train, руб:", mean_train_cost)


# S06-C05
def error_metrics(actual, predictions):
    absolute_sum = 0.0
    squared_sum = 0.0
    for i in range(len(actual)):
        error = predictions[i] - actual[i]
        absolute_sum = absolute_sum + abs(error)
        squared_sum = squared_sum + error**2
    mae = absolute_sum / len(actual)
    mse = squared_sum / len(actual)
    rmse = sqrt(mse)
    return float(mae), rmse


line_train_predictions = line_model.predict(distance_features)
line_errors = []
for i in range(len(train_y)):
    line_errors.append(line_train_predictions[i] - train_y[i])
plt.scatter(train_x, line_errors, c=train_peak, cmap="coolwarm")
plt.axhline(0, color="black")
plt.xlabel("Расстояние, км")
plt.ylabel("Ошибка цены: прогноз минус факт, руб")
plt.show()


# S06-C06
peak_features = []
for i in range(len(train_x)):
    peak_features.append([train_x[i], train_peak[i]])
peak_model = LinearRegression()
peak_model.fit(peak_features, train_y)
print("Коэффициенты и b:", peak_model.coef_, peak_model.intercept_)


# S06-C07
peak_train_predictions = peak_model.predict(peak_features)
peak_errors = []
for i in range(len(train_y)):
    peak_errors.append(peak_train_predictions[i] - train_y[i])
plt.scatter(train_x, peak_errors, c=train_peak, cmap="coolwarm")
plt.axhline(0, color="black")
plt.xlabel("Расстояние, км")
plt.ylabel("Ошибка цены после учёта часа пик, руб")
plt.show()


# S06-C08
shape_features = []
for i in range(len(train_x)):
    distance = train_x[i]
    shape_features.append([distance, distance**2, train_peak[i]])
shape_model = LinearRegression()
shape_model.fit(shape_features, train_y)
print("Коэффициенты и b:", shape_model.coef_, shape_model.intercept_)


# S06-C09
test_y = []
test_distance = []
test_peak = []
test_shape = []
constant_predictions = []
for row in test_rows:
    distance = float(row["distance_km"])
    peak = int(row["peak"])
    test_y.append(float(row["cost_rub"]))
    test_distance.append([distance])
    test_peak.append([distance, peak])
    test_shape.append([distance, distance**2, peak])
    constant_predictions.append(mean_train_cost)
line_test_predictions = line_model.predict(test_distance)
peak_test_predictions = peak_model.predict(test_peak)
shape_test_predictions = shape_model.predict(test_shape)
print("Среднее, MAE и RMSE:", error_metrics(test_y, constant_predictions))
print("Расстояние, MAE и RMSE:", error_metrics(test_y, line_test_predictions))
print("Расстояние и пик:", error_metrics(test_y, peak_test_predictions))
print("Расстояние, квадрат и пик:", error_metrics(test_y, shape_test_predictions))


# S06-C10
new_trip = [[5.0, 25.0, 1]]
new_prediction = shape_model.predict(new_trip)[0]
print("Прогноз, руб:", round(new_prediction))
print("Запас до 700 рублей:", 700 - new_prediction)


# S06-C11
with (data_dir / "taxi-extra-trip.csv").open() as file:
    extra_rows = list(csv.DictReader(file))
extra_distance = float(extra_rows[0]["distance_km"])
extra_peak = int(extra_rows[0]["peak"])
extra_cost = float(extra_rows[0]["cost_rub"])
changed_features = shape_features + [[extra_distance, extra_distance**2, extra_peak]]
changed_y = train_y + [extra_cost]
changed_model = LinearRegression()
changed_model.fit(changed_features, changed_y)
print("Цена поездки 5 км до и после:", new_prediction, changed_model.predict(new_trip)[0])


# S06-C12
old_changed_predictions = shape_model.predict(changed_features)
new_changed_predictions = changed_model.predict(changed_features)
print("49 train, до:", error_metrics(changed_y, old_changed_predictions))
print("49 train, после:", error_metrics(changed_y, new_changed_predictions))
changed_test_predictions = changed_model.predict(test_shape)
print("16 test, до:", error_metrics(test_y, shape_test_predictions))
print("16 test, после:", error_metrics(test_y, changed_test_predictions))


# S06-C13
quiet_x = []
quiet_y = []
for i in range(len(train_x)):
    if train_peak[i] == 0:
        quiet_x.append(train_x[i])
        quiet_y.append(train_y[i])
line_x = []
line_features = []
for i in range(1, 25):
    distance = i / 2
    line_x.append(distance)
    line_features.append([distance, distance**2, 0])
plt.scatter(quiet_x, quiet_y, label="Train вне часа пик")
plt.scatter([extra_distance], [extra_cost], color="red", label="Отдельный чек 1800 руб")
plt.plot(line_x, shape_model.predict(line_features), label="До")
plt.plot(line_x, changed_model.predict(line_features), label="После")
plt.xlabel("Расстояние, км")
plt.ylabel("Цена при пик=0, руб")
plt.legend()
plt.show()


# S06-C14
print("A: в чеке 180000 копеек, в таблице единица рубли")
print("B: чек подтверждает 1800 рублей за одну поездку")
print("C: модель даёт цену поездки 5 км в час пик")
