# Thompson Sampling Module

Библиотека для онлайн-принятия решений в условиях неопределённости. Задача: из набора вариантов (плечей) выбрать наиболее перспективный с учётом контекста — и продолжать обучаться на каждом поступающем результате.

Подходит для задач маршрутизации, ранжирования и рекомендаций, где награда за решение становится известна позже и оптимальный выбор зависит от контекста.

**Репозиторий:** https://github.com/Ulandaev/thompson_sampling_module

---

## Содержание

- [Быстрый старт](#быстрый-старт)
- [Модели](#модели)
- [Конфигурация](#конфигурация)
- [Архитектура и модули](#архитектура-и-модули)
- [Симуляция: пример с маршрутизацией тикетов](#симуляция-пример-с-маршрутизацией-тикетов)
- [Тесты](#тесты)
- [Сохранение состояния](#сохранение-состояния)
- [ToDo](#todo)

---

## Быстрый старт

**Требования:** Python 3.11+, [Poetry](https://python-poetry.org/)

```bash
git clone https://github.com/Ulandaev/thompson_sampling_module
cd thompson_sampling_module
poetry install
```

```python
from ts_module import TSEngine

engine = TSEngine.from_yaml("examples/ticket_routing/config.yaml")

# Принять решение — двжиок вернёт лучшее плечо по текущему постериору
result = engine.decide(context={"ticket_category": "billing", "client_tier": "premium"})
print(result.recommended_arm)   # например: "agent_maria"
print(result.arm_scores)        # апостериорные оценки всех плечей

# Отправить фидбек — движок обновит постериор
engine.feedback(result.session_id, [{"name": "fcr", "value": 1.0}])
```

Это полный цикл. Состояние хранится внутри движка — внешнее управление не нужно.

---

## Модели

Реализованы три модели. Выбор происходит автоматически по типу награды и контексту (`model_type: auto`), либо задаётся явно в конфиге.

### Beta TS

**Когда использовать:** бинарная награда (успех / неуспех), категориальный или отсутствующий контекст.

Для каждого плеча и каждого контекстного ключа хранится независимое распределение `Beta(α, β)`:

- При успехе: `α += 1`
- При неуспехе: `β += 1`
- Сэмпл: `θ ~ Beta(α, β)` — вероятность успеха для данного плеча

Модель не делает обобщений между контекстными ключами — каждая пара (плечо, контекст) обучается независимо. Это и сила (нет смешения), и ограничение (нужно больше данных на каждую комбинацию).

```yaml
hyperparams:
  model_type: beta
  beta:
    alpha_init: 1.0       # начальный prior (равновероятный)
    beta_init: 1.0
    cold_start_pulls: 30  # исследовать каждое плечо минимум 30 раз
    min_exploration: 0.03 # минимальный трафик на каждое плечо
```

### Logistic TS

**Когда использовать:** бинарная награда + непрерывные или категориальные признаки.

Модель хранит байесовскую линейную регрессию с сигмоид-активацией. Для каждого плеча обучается вектор весов `w` по всем признакам контекста:

```
p = σ(wᵀx)
```

Постериор: `w ~ N(m, A⁻¹)`. Обновление — онлайн-IRLS (итерационно взвешенный метод наименьших квадратов), ковариация обновляется по Sherman-Morrison, среднее — по шагу Ньютона:

```
m_new = m + A_inv_new · (r − p) · x
```

Это важное отличие от наивной реализации: при `r = 0` (неуспех) среднее убывает, а не остаётся на месте. Веса общие для всех контекстных значений — модель обобщает знания между ними.

```yaml
context:
  mode: features
  features:
    - {name: ticket_category, type: categorical, categories: [billing, tech, complaint]}
    - {name: time_of_day,     type: continuous}

hyperparams:
  model_type: logistic
  logistic:
    prior_variance: 1.0     # σ² приора: больше → менее информативный, учится быстрее
    regularization: 0.01    # λ на диагонали A для численной устойчивости
    feature_scaling: false  # стандартизация непрерывных признаков (z-score)
```

> **Заметка по `feature_scaling`:** при `true` масштабирование подстраивается лениво под первый наблюдённый контекст. Если он нетипичен (выброс), знак веса для признака может инвертироваться. Для признаков с известным диапазоном (например, `time_of_day` в часах) безопаснее выставить `false`.

### Linear TS

**Когда использовать:** непрерывная награда (выручка, NPS, длительность, любое вещественное значение).

Та же байесовская линейная регрессия, без сигмоиды. Предсказание: `ŷ = wᵀx`. Ключевое отличие от LogisticModel — коэффициент масштабирования точности `1/σ²_noise` применяется одинаково и к обновлению ковариации, и к аккумуляции вектора `b`:

```
A_new = A + (1/σ²) · xxᵀ
b_new = b + (1/σ²) · r · x
w_map = A_inv · b → истинному среднему при n → ∞
```

```yaml
hyperparams:
  model_type: linear
  linear:
    prior_variance: 1.0
    noise_variance: 1.0   # σ²: меньше → обучается быстрее, агрессивнее
    feature_scaling: true
```

### Автоматический выбор модели

| `reward.type` | `context.mode` | Результат |
|---|---|---|
| `binary` | `none` или `categorical` | **Beta TS** |
| `binary` | `features` | **Logistic TS** |
| `continuous` | любой | **Linear TS** |

---

## Конфигурация

Всё поведение движка описывается одним YAML-файлом. Менять модель, добавлять плечи, переключать метрику — без правок кода.

```yaml
id: my_module
name: "Пример конфигурации"

# Варианты для выбора (плечи)
arms:
  - {id: option_a, name: "Вариант А", metadata: {cost: 100}}
  - {id: option_b, name: "Вариант Б", metadata: {cost: 120}}
  - {id: option_c, name: "Вариант В", metadata: {cost: 90}}

# Контекст, доступный в момент принятия решения
context:
  mode: features             # none | categorical | features
  features:
    - {name: segment,   type: categorical, categories: [new, returning, vip]}
    - {name: score,     type: continuous}

# Структура награды
reward:
  type: binary               # binary | continuous
  signals:
    - {name: conversion, weight:  1.0, timeout_hours: 48, default_on_timeout: 0.0}
    - {name: refund,     weight: -1.0, timeout_hours: 72, default_on_timeout: 0.0}
  aggregation: weighted_sum

# Целевая функция
objective:
  type: max_probability      # max_probability | max_expected_revenue | custom

# Бизнес-ограничения
constraints:
  - {type: min_traffic, value: 0.05}  # каждое плечо получает не менее 5% трафика

# Гиперпараметры
hyperparams:
  model_type: auto
  min_exploration: 0.03
```

---

## Архитектура и модули

```
ts_module/
├── config/
│   ├── schema.py          # Pydantic v2 схемы всей конфигурации
│   └── loader.py          # load_from_yaml(), load_from_dict()
└── core/
    ├── engine.py          # TSEngine — главный публичный интерфейс
    ├── preprocessing.py   # FeatureProcessor: dict → numpy-вектор
    ├── feedback.py        # FeedbackAggregator: сигналы → скалярная награда
    ├── objective.py       # ObjectiveFunction: θ → оценка плеча
    ├── constraints.py     # ConstraintEngine: capacity, min_traffic
    ├── models/
    │   ├── base.py        # BaseModel (ABC): sample, update, get_distribution
    │   ├── beta.py        # BetaModel
    │   ├── logistic.py    # _BayesianLinearBase + LogisticModel
    │   └── linear.py      # LinearModel
    └── state/
        ├── base.py        # StateStore ABC, SessionData
        └── memory.py      # InMemoryStateStore
```

### TSEngine (`core/engine.py`)

Оркестрирует весь цикл принятия решений:

1. **`decide(context, seed)`** — фильтрует плечи через `ConstraintEngine`, сэмплирует θ для каждого плеча через модель, применяет `ObjectiveFunction`, возвращает `DecisionResult`.
2. **`feedback(session_id, signals)`** — передаёт сигналы в `FeedbackAggregator`, получает агрегированную награду, вызывает `model.update()`.

### FeatureProcessor (`core/preprocessing.py`)

Преобразует словарь контекста в фиксированный numpy-вектор:
- Категориальные признаки → one-hot encoding (по списку `categories` из конфига)
- Непрерывные признаки → опциональная z-score стандартизация
- Неизвестная категория или пропущенный признак → нули (без ошибок)
- Масштабирование подстраивается лениво при первом вызове `transform()`

### FeedbackAggregator (`core/feedback.py`)

Агрегирует несколько сигналов награды в один скаляр по формуле взвешенной суммы. Поддерживает таймауты: если сигнал не пришёл в течение `timeout_hours`, подставляется `default_on_timeout`.

### ConstraintEngine (`core/constraints.py`)

Применяет бизнес-ограничения поверх байесовских оценок:
- `min_traffic` — гарантирует минимальную долю трафика каждому плечу (exploration floor)
- `capacity` — исключает перегруженные плечи из кандидатов
- `eligibility` — фильтрация по произвольному полю контекста

### ObjectiveFunction (`core/objective.py`)

Преобразует апостериорный сэмпл θ в финальную оценку плеча. Варианты: `max_probability`, `max_expected_revenue`, `max_roi`, `min_cost_per_success`, `custom` (произвольная формула).

---

## Симуляция: пример с маршрутизацией тикетов

В папке `examples/ticket_routing/` реализован полный пример: 5 операторов колл-центра, 4 категории тикетов, бинарная метрика FCR (First Contact Resolution).

### Запуск

```bash
# Вывод таблиц в консоль
poetry run python examples/ticket_routing/simulate.py

# Сохранить графики в папку results/
poetry run python examples/ticket_routing/simulate.py --save-results

# Показать графики интерактивно
poetry run python examples/ticket_routing/simulate.py --show-plots

# Изменить количество тикетов или сид
poetry run python examples/ticket_routing/simulate.py --tickets 3000 --seed 7
```

### Два эксперимента

**Эксперимент 1 — без признака времени суток.**
Сравниваются Round-robin, Beta TS и Logistic TS на одних и тех же данных. Цель — изолировать *алгоритмическое* превосходство Logistic TS над Beta: Logistic обобщает веса между всеми контекстными значениями, тогда как Beta обучает каждую пару (агент, категория) независимо.

**Эксперимент 2 — с признаком `time_of_day`.**
Каждый агент работает эффективнее в определённое время суток (скрытые бонусы FCR). Сравниваются Beta TS, Logistic без временного признака и Logistic с полным набором признаков. Показывает *признаковое* преимущество от расширения контекста.

### Генерируемые графики

| Файл | Содержание |
|---|---|
| `results/fcr_exp1.png` | Накопленный FCR — Эксперимент 1 |
| `results/regret_exp1.png` | Накопленное сожаление vs Oracle — Эксперимент 1 |
| `results/traffic_exp1.png` | Распределение трафика по агентам — Эксперимент 1 |
| `results/fcr_exp2.png` | Накопленный FCR — Эксперимент 2 |
| `results/regret_exp2.png` | Накопленное сожаление vs Oracle — Эксперимент 2 |
| `results/traffic_exp2_no.png` | Трафик — Logistic без временного признака |
| `results/traffic_exp2_full.png` | Трафик — Logistic с полным набором признаков |

### Конфиги для симуляции

| Файл | Модель | Контекст |
|---|---|---|
| `config.yaml` | Beta TS | Категория тикета + уровень клиента |
| `config_contextual.yaml` | Logistic TS | Те же признаки + one-hot encoding |
| `config_contextual_full.yaml` | Logistic TS | + `time_of_day` как непрерывный признак |

---

## Тесты

```bash
# Все тесты
poetry run pytest

# По группам
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/simulation/
```

75 тестов, все проходят. Покрытие:

| Группа | Тестов | Что проверяется |
|---|---|---|
| `unit/test_config.py` | 10 | Валидация конфига, авто-выбор модели, граничные случаи |
| `unit/test_beta_model.py` | 8 | BetaModel: сэмплинг, обновление, сериализация |
| `unit/test_logistic_model.py` | 10 | LogisticModel: IRLS, сходимость, state roundtrip |
| `unit/test_linear_model.py` | 6 | LinearModel: сходимость, детерминизм, сериализация |
| `unit/test_preprocessing.py` | 8 | FeatureProcessor: one-hot, scaling, lazy fit |
| `unit/test_constraints.py` | 8 | ConstraintEngine: capacity, min_traffic floor |
| `unit/test_feedback.py` | 8 | FeedbackAggregator: агрегация, таймауты |
| `unit/test_objective.py` | 8 | ObjectiveFunction: все типы целевых функций |
| `integration/test_phase2_compatibility.py` | 3 | Движок корректно создаёт нужную модель по конфигу |
| `simulation/test_ticket_routing.py` | 6 | Beta TS сходится быстрее round-robin |
| `simulation/test_contextual_routing.py` | 3 | Logistic TS обнаруживает временной паттерн |

---

## Сохранение состояния

Движок хранит состояние модели в памяти. Для сохранения и восстановления:

```python
# Сохранить
state = engine.get_model_state()
import json
with open("model_state.json", "w") as f:
    json.dump(state, f)

# Восстановить
with open("model_state.json") as f:
    state = json.load(f)
engine.load_model_state(state)
```

`get_state()` возвращает обычный словарь (numpy-массивы сериализованы в списки), корректно проходит через JSON.

---

## ToDo

Известные ограничения и запланированные доработки:

- **Персистентное хранилище** — только in-memory. Перезапуск сервиса сбрасывает весь накопленный постериор. Нужен бэкенд на SQLite или Redis.
- **REST API** — зависимости FastAPI/Uvicorn уже прописаны в `pyproject.toml`, слой API не реализован.
- **Пакетные обновления** — `feedback()` принимает одно наблюдение за раз. Инициализация модели из исторических логов требует цикла.
- **Дрейф данных** — у BetaModel есть параметр `forgetting_rate`, у Logistic и Linear аналога нет. Для долгоживущих деплоев нужен механизм забывания или детекции концептуального дрейфа.
- **`composite` тип награды** — частично описан в схеме, в движке не реализован.
- **Многоцелевая оптимизация** — сейчас одна целевая функция на модуль. Pareto-фронт между несколькими метриками не поддерживается.
