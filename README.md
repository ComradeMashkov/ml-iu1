# Машинное обучение · ИУ-1 (САУ ЛА)

Материалы инженерного курса Машкова И.И. для 6-го курса кафедры ИУ-1 МГТУ им. Н.Э. Баумана,
осенний семестр 2026.

Курс связывает честную оценку моделей на сенсорных данных с бортовым выводом:
**scikit-learn → PyTorch → ONNX Runtime C++ → Docker → GitHub Actions**.
Сайт и reveal.js-слайды собраны на Quarto; графики генерируются исполняемым Python-кодом.

## Структура

```text
.
├── _quarto.yml                    # сайт, навигация и общие метаданные
├── index.qmd                      # позиционирование и календарь
├── syllabus.qmd                   # результаты обучения, стек и оценивание
├── lectures/
│   ├── index.qmd                  # карта девяти лекций
│   ├── _template.qmd              # непубликуемый шаблон reveal.js
│   └── L00-engineering-ml.qmd     # вводная лекция
├── seminars/index.qmd             # 17 лабораторных итераций
├── project/
│   ├── index.qmd                  # протокол проекта и рубрика
│   └── _project-card-template.md  # шаблон постановки проекта
├── assets/
│   ├── figstyle.py                # единый matplotlib-стиль
│   ├── theme/custom.scss          # тема сайта
│   ├── theme/slides.scss          # тема слайдов
│   └── references.bib             # библиография
├── pyproject.toml                 # диапазоны Python-зависимостей
├── uv.lock                        # точные версии и хеши
└── .github/workflows/publish.yml  # воспроизводимая сборка GitHub Pages
```

## Локальная сборка

Нужны [Quarto](https://quarto.org) и [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
make render
```

Предпросмотр:

```bash
make preview
```

`QUARTO_PYTHON` автоматически указывает на Python из `.venv`, поэтому Jupyter-ячейки исполняются
в том же окружении, которое зафиксировано в `uv.lock`.

## Как добавить лекцию

1. Скопировать `lectures/_template.qmd` в файл вида `lectures/L01-validation.qmd`.
2. Сначала зафиксировать 2–3 измеримых результата обучения и артефакт следующей лаборатории.
3. Добавить материал, заметки докладчика и воспроизводимые графики.
4. Добавить ссылку в `lectures/index.qmd` и выполнить `make render`.

## Публикация

Workflow устанавливает Python 3.12, uv, зависимости из lock-файла и Quarto, затем собирает `_site`.
Основная публичная версия публикуется через GitHub Pages:
<https://comrademashkov.github.io/ml-iu1/>.

Публичная production-версия также поддерживается через Sites. Команда `make sites-dist` собирает
Quarto и готовит статический Worker-артефакт; `.openai/hosting.json` связывает каталог с существующим сайтом.

---

© 2026 · Машков И.И. · кафедра ИУ-1, МГТУ им. Н.Э. Баумана.
