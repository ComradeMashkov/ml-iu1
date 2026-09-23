"""S1: learn one temperature threshold using ordinary Python loops."""

# %% 1. Данные: один запуск — одно число и один ответ
# Все числа придуманы для занятия. Температура дана в градусах Цельсия.
train_temperature = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
train_target = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]

print("Первый запуск:", train_temperature[0], train_target[0])
print("Число запусков:", len(train_temperature))

# %% 2. Правило для одного запуска
temperature = 58
threshold = 50

if temperature >= threshold:
    prediction = 1
else:
    prediction = 0

print("Температура:", temperature, "Прогноз:", prediction)

# %% 3. Ошибки одного правила на обучении
threshold = 50
errors = 0

for i in range(len(train_temperature)):
    if train_temperature[i] >= threshold:
        prediction = 1
    else:
        prediction = 0

    if prediction != train_target[i]:
        errors = errors + 1

print("Порог:", threshold, "Ошибок:", errors)

# %% 4. Обучение: сравнение четырёх порогов
candidates = [40, 50, 60, 70]
best_threshold = candidates[0]
best_errors = len(train_temperature) + 1
candidate_errors = []

for threshold in candidates:
    errors = 0
    for i in range(len(train_temperature)):
        if train_temperature[i] >= threshold:
            prediction = 1
        else:
            prediction = 0

        if prediction != train_target[i]:
            errors = errors + 1

    candidate_errors.append(errors)
    print("Порог:", threshold, "Ошибок:", errors)
    if errors < best_errors:
        best_errors = errors
        best_threshold = threshold

print("Выбранный порог:", best_threshold)

# %% 5. Прогноз для нового запуска
new_temperature = 58

if new_temperature >= best_threshold:
    new_prediction = 1
else:
    new_prediction = 0

print("Новый запуск:", new_temperature, "Прогноз:", new_prediction)

# %% 6. Проверка на шести других запусках
# Эти строки не участвовали в выборе порога.
check_temperature = [38, 48, 58, 62, 68, 78]
check_target = [0, 1, 0, 0, 1, 1]
check_prediction = []
correct = 0
always_zero_correct = 0

for i in range(len(check_temperature)):
    if check_temperature[i] >= best_threshold:
        prediction = 1
    else:
        prediction = 0
    check_prediction.append(prediction)

    if prediction == check_target[i]:
        correct = correct + 1
    if check_target[i] == 0:
        always_zero_correct = always_zero_correct + 1

    print(check_temperature[i], "ответ:", check_target[i], "прогноз:", prediction)

accuracy = correct / len(check_target)
baseline_accuracy = always_zero_correct / len(check_target)
print("Верных ответов:", correct, "из", len(check_target))
print("Доля верных ответов:", accuracy)
print("Всегда отвечать 0:", baseline_accuracy)
