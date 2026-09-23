"""S3: последовательный код занятия. Запуск из starter: python lessons/s03.py."""

# %% Подготовлено: импорты
from math import exp, log

import matplotlib.pyplot as plt

# %% 1. Сообщение и один признак
messages = [
    "See you at six",
    "Are you free tomorrow",
    "Win a free ticket",
    "Free prize win now",
]
y = [0, 0, 1, 1]


def count_words(message):
    count = 0
    for word in message.lower().split():
        word = word.strip(".,!?;:")
        if word in ["free", "win", "prize"]:
            count = count + 1
    return count


x = []
for message in messages:
    x.append(count_words(message))
print("x =", x)
print("y =", y)


# %% 2. Прогноз при известных параметрах
def probability(x_value, w, b):
    z = w * x_value + b
    return 1 / (1 + exp(-z))


w = 0.0
b = 0.0
for i in range(len(x)):
    p = probability(x[i], w, b)
    print("x =", x[i], "y =", y[i], "p =", round(p, 3))


# %% 3. Средняя потеря
def mean_loss(x, y, w, b):
    total = 0.0
    for i in range(len(x)):
        p = probability(x[i], w, b)
        loss = -y[i] * log(p) - (1 - y[i]) * log(1 - p)
        total = total + loss
    return total / len(x)


print("Потеря до обучения:", round(mean_loss(x, y, w, b), 6))


# %% 4. Один шаг обучения
def gradients(x, y, w, b):
    dw = 0.0
    db = 0.0
    for i in range(len(x)):
        p = probability(x[i], w, b)
        error = p - y[i]
        dw = dw + error * x[i]
        db = db + error
    return dw / len(x), db / len(x)


dw, db = gradients(x, y, w, b)
learning_rate = 1.0
w = w - learning_rate * dw
b = b - learning_rate * db
print("Градиент:", dw, db)
print("После шага:", w, b)
print("Потеря после шага:", round(mean_loss(x, y, w, b), 6))

# %% 5. Повторение одного и того же шага
w = 0.0
b = 0.0
loss_history = [mean_loss(x, y, w, b)]
for step in range(200):
    dw, db = gradients(x, y, w, b)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    loss_history.append(mean_loss(x, y, w, b))

print("После 200 шагов:", round(w, 3), round(b, 3))
print("Потеря:", round(loss_history[-1], 6))
for i in range(len(x)):
    print("x =", x[i], "p =", round(probability(x[i], w, b), 3))

# %% Подготовлено: график вычисленной истории
plt.plot(loss_history)
plt.xlabel("Число выполненных шагов")
plt.ylabel("Средняя потеря на четырёх сообщениях")
plt.grid(alpha=0.2)
plt.show()

# %% 6. Новое сообщение
new_message = "You win a prize today"
new_x = count_words(new_message)
new_p = probability(new_x, w, b)
print("Число слов:", new_x, "Оценка спама:", round(new_p, 3))

ordinary_message = "Are you free to win a prize at our family quiz"
ordinary_x = count_words(ordinary_message)
ordinary_p = probability(ordinary_x, w, b)
print("Семейная викторина:", ordinary_x, round(ordinary_p, 3))
