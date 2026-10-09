"""Время доставки: обучение и проверка. Run from starter or the course root. All data are synthetic."""

# S05-C01
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


# S05-C02
with (data_dir / "delivery-orders-synthetic.csv").open() as file:
    rows = list(csv.DictReader(file))
train_rows = []
test_rows = []
for row in rows:
    if row["split"] == "train":
        train_rows.append(row)
    else:
        test_rows.append(row)
train_x = []
train_y = []
for row in train_rows:
    train_x.append(float(row["distance_km"]))
    train_y.append(float(row["delivery_minutes"]))
print("Train:", len(train_rows), "Test:", len(test_rows))


# S05-C03
plt.scatter(train_x, train_y)
plt.xlabel("Расстояние, км")
plt.ylabel("Время, мин")
plt.title("36 учебных заказов")
plt.grid(alpha=0.2)
plt.show()


# S05-C04
total_time = 0.0
for time_value in train_y:
    total_time = total_time + time_value
mean_train_time = total_time / len(train_y)
print("Постоянный прогноз, мин:", mean_train_time)


# S05-C05
tiny_x = train_x[:3]
tiny_y = train_y[:3]
w = 0.0
b = 20.0
for i in range(len(tiny_x)):
    prediction = w * tiny_x[i] + b
    error = prediction - tiny_y[i]
    print(tiny_x[i], tiny_y[i], prediction, error, error * tiny_x[i], error**2)


# S05-C06
def mean_squared_error(x_values, y_values, w, b):
    total_squared_error = 0.0
    for i in range(len(x_values)):
        prediction = w * x_values[i] + b
        error = prediction - y_values[i]
        total_squared_error = total_squared_error + error**2
    return total_squared_error / len(x_values)


print("MSE до шага:", mean_squared_error(tiny_x, tiny_y, 0, 20))


# S05-C07
def gradients(x_values, y_values, w, b):
    sum_error_x = 0.0
    sum_error = 0.0
    for i in range(len(x_values)):
        prediction = w * x_values[i] + b
        error = prediction - y_values[i]
        sum_error_x = sum_error_x + error * x_values[i]
        sum_error = sum_error + error
    dw = 2 * sum_error_x / len(x_values)
    db = 2 * sum_error / len(x_values)
    return dw, db


print("Градиент:", gradients(tiny_x, tiny_y, 0, 20))


# S05-C08
w = 0.0
b = 20.0
learning_rate = 0.1
before = mean_squared_error(tiny_x, tiny_y, w, b)
dw, db = gradients(tiny_x, tiny_y, w, b)
w = w - learning_rate * dw
b = b - learning_rate * db
after = mean_squared_error(tiny_x, tiny_y, w, b)
print("w, b:", w, b)
print("MSE до и после:", before, after)


# S05-C09
w = 0.0
b = 0.0
learning_rate = 0.03
loss_history = [mean_squared_error(train_x, train_y, w, b)]
for step in range(1500):
    dw, db = gradients(train_x, train_y, w, b)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    loss_history.append(mean_squared_error(train_x, train_y, w, b))
print("Параметры:", w, b)
print("MSE train:", loss_history[-1])


# S05-C10
plt.plot(loss_history)
plt.xlabel("Шаг обучения")
plt.ylabel("MSE train, мин²")
plt.yscale("log")
plt.grid(alpha=0.2)
plt.show()


# S05-C11
rate_results = []
for rate in [0.003, 0.03, 0.1]:
    trial_w = 0.0
    trial_b = 0.0
    for step in range(30):
        dw, db = gradients(train_x, train_y, trial_w, trial_b)
        trial_w = trial_w - rate * dw
        trial_b = trial_b - rate * db
    trial_loss = mean_squared_error(train_x, train_y, trial_w, trial_b)
    rate_results.append([rate, trial_loss])
    print("Шаг и MSE после 30 итераций:", rate, trial_loss)


# S05-C12
line_x = [0.5, 6.0]
line_y = [w * line_x[0] + b, w * line_x[1] + b]
plt.scatter(train_x, train_y, label="36 train")
plt.plot(line_x, line_y, label="Линейная модель")
plt.axhline(mean_train_time, color="orange", label="Среднее train")
plt.xlabel("Расстояние, км")
plt.ylabel("Время, мин")
plt.legend()
plt.show()


# S05-C13
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


test_y = []
model_predictions = []
baseline_predictions = []
for row in test_rows:
    distance = float(row["distance_km"])
    test_y.append(float(row["delivery_minutes"]))
    model_predictions.append(w * distance + b)
    baseline_predictions.append(mean_train_time)
print("Среднее, MAE и RMSE:", error_metrics(test_y, baseline_predictions))
print("Прямая, MAE и RMSE:", error_metrics(test_y, model_predictions))


# S05-C14
new_distance = 2.0
new_prediction = w * new_distance + b
print("Прогноз, мин:", round(new_prediction, 1))
