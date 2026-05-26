# TS Module — Контекст проекта

## Что это

Универсальный модуль принятия решений на основе сэмплинга Томпсона (Thompson Sampling).
Продукт для платформы BPMSoft — low-code системы оптимизации бизнес-процессов.

Модуль встраивается в любой бизнес-процесс как «умный узел выбора»: вместо жёсткого
правила он учится на результатах и со временем выбирает лучший вариант для каждой ситуации.

## Бизнес-контекст

Заказчик — BPMSoft. Их клиенты — компании с CRM, service desk, маркетингом, продажами.
В каждом процессе есть точка выбора: кому назначить тикет, что предложить клиенту,
по какому каналу отправить. Сейчас это решается правилами или интуицией.

Наш модуль делает эти точки адаптивными. Клиент не знает математику — он видит:
«мои тикеты назначаются умнее, FCR вырос на 15%».

## Принцип работы: сэмплинг Томпсона

Для каждой «руки» (варианта) хранится Beta-распределение Beta(α, β).
α — число успехов, β — число неудач.

При каждом решении:
1. Сэмплировать θk ~ Beta(αk, βk) для каждой руки k
2. Выбрать руку k* = argmax(objective(θk, arm_metadata, context))
3. Наблюдать результат
4. Обновить: успех → αk* += reward, неудача → βk* += |reward|

Ключевое свойство: баланс exploration/exploitation без ручной настройки.
Regret растёт логарифмически, а не линейно как в A/B-тестах.

## Архитектура модуля: три слоя

### Слой 1 — Ядро (Core Engine)
Неизменный для всех задач. Реализует математику TS.

### Слой 2 — Конфигурация
Определяет поведение для конкретной задачи через 5 вопросов:
1. Arms: что выбираем?
2. Context: что знаем в момент решения?
3. Reward: что считать успехом?
4. Objective: как сравнивать варианты?
5. Constraints: какие бизнес-ограничения?

### Слой 3 — Интеграция (Phase 3)
API, async feedback, persistence. В Phase 1-2 не реализуется.

## Варианты модели (model_type)

| Модель | Тип reward | Тип контекста | Состояние |
|--------|-----------|---------------|-----------|
| Beta TS | binary | none / categorical | Beta(α,β) per arm×context |
| Logistic TS | binary | continuous features | N(μ,Σ) per arm |
| Linear TS | continuous | continuous features | N(μ,Σ) per arm |
| Composite | binary × continuous | any | Logistic × Linear |

**Выбор модели (auto):**
- binary + none/categorical → Beta TS
- binary + features → Logistic TS (Phase 2)
- continuous + any → Linear TS (Phase 2)

## Компоненты конфигурации

### Arms
```yaml
arms:
  - id: agent_maria
    name: "Мария С."
    metadata:
      capacity: 8        # max concurrent tickets
      cost_per_hour: 150
```

### Context Config
```yaml
context:
  mode: none | categorical | features
  features:
    - name: ticket_category
      type: categorical   # | continuous
```

Context key для Beta TS (categorical mode):
"billing__enterprise" → отдельное Beta(α,β) для каждой комбинации значений

### Reward Config
```yaml
reward:
  type: binary | continuous | composite
  signals:
    - name: fcr
      weight: 1.0
      timeout_hours: 72
      default_on_timeout: 0.0
    - name: reopen
      weight: -0.5       # штраф
      timeout_hours: 72
      default_on_timeout: 0.0
  aggregation: weighted_sum | first_positive | max
```

Итоговый reward = сумма (signal.value × signal.weight) по всем пришедшим сигналам.
Обновление модели: если reward > 0 → α += reward, если reward <= 0 → β += |reward|

### Objective Function
```yaml
objective:
  type: max_probability        # argmax(θk)
       | max_expected_revenue  # argmax(θk × arm.metadata.price)
       | max_roi               # argmax((θk×value - cost) / cost)
       | min_cost_per_success  # argmin(cost / θk)
       | custom
  formula: "theta * arm.price * context.clv_score"  # для custom
```

Переменные в формуле:
- `theta` — сэмплированная вероятность
- `arm.{field}` — поле из arm.metadata
- `context.{field}` — поле из контекста запроса

### Constraints
```yaml
constraints:
  - type: capacity            # исключить перегруженные arms
    arm_field: metadata.capacity
  - type: min_traffic         # гарантировать минимальный трафик
    value: 0.03               # 3% на каждую руку
  - type: eligibility         # предварительная фильтрация
    condition: "arm.metadata.active == true"
```

### Hyperparameters
```yaml
hyperparams:
  model_type: auto | beta | logistic | linear
  beta:
    alpha_init: 1.0           # начальное alpha (оптимистичный prior если > 1)
    beta_init: 1.0            # начальное beta
    forgetting_rate: 0.97     # γ: умножать α,β на γ раз в сутки (Phase 4)
    min_exploration: 0.03     # минимальная доля трафика на любую руку
    cold_start_pulls: 30      # мин. событий до "доверия" оценке
  update_mode: realtime | batch
```

## Интерфейс Engine (Phase 1)

```python
# Запрос решения
result = engine.decide(
    context={"ticket_category": "complaint", "client_tier": "enterprise"},
    eligible_arms=["agent_maria", "agent_ivan"],  # None = все arms из конфига
    session_id="ticket_88421",                     # None = генерируется автоматически
    seed=None                                       # int для детерминизма в тестах
)
# DecisionResult:
#   session_id: str
#   recommended_arm: str
#   confidence: float  (max_score - second_max_score)
#   arm_scores: dict[str, float]
#   model_state_snapshot: dict  (α,β в момент решения — для аудита)

# Обратная связь
result = engine.feedback(
    session_id="ticket_88421",
    signals=[
        {"name": "fcr",    "value": 1.0},
        {"name": "reopen", "value": 0.0}
    ]
)
# UpdateResult:
#   session_id: str
#   arm_updated: str
#   reward_applied: float
#   new_distribution: dict  (новые α,β)
```

## Симуляция маршрутизации тикетов

Основной сценарий для проверки алгоритма в Phase 1.

### Истинные FCR (известны симулятору, скрыты от модели)
```python
TRUE_FCR = {
    "agent_maria":  {"billing":0.72,"tech":0.68,"complaint":0.91,"onboard":0.70},
    "agent_ivan":   {"billing":0.81,"tech":0.74,"complaint":0.61,"onboard":0.79},
    "agent_alexey": {"billing":0.65,"tech":0.88,"complaint":0.59,"onboard":0.82},
    "agent_olga":   {"billing":0.78,"tech":0.71,"complaint":0.74,"onboard":0.93},
    "agent_dmitry": {"billing":0.70,"tech":0.65,"complaint":0.67,"onboard":0.68},
}
ORACLE = {  # максимально достижимый FCR
    "billing": ("agent_ivan", 0.81),
    "tech": ("agent_alexey", 0.88),
    "complaint": ("agent_maria", 0.91),
    "onboard": ("agent_olga", 0.93),
}
```

### Три стратегии для сравнения
1. **Round-robin** — равномерное случайное назначение (baseline)
2. **TS (наш модуль)** — основной метод
3. **Oracle** — всегда назначает лучшего агента (теоретический максимум)

### Метрики
- Накопленный FCR по тикетам
- Regret = Oracle_FCR - TS_FCR (накопленный)
- Распределение трафика по агентам в динамике
- К какому тикету TS устойчиво выбирает правильного агента (per category)

## Фазы разработки

### Phase 1 — Core + Simulation (текущая)
- Beta TS (Regular + Categorical)
- In-memory state store
- TSEngine с синхронным feedback
- Конфигурация (Pydantic + YAML)
- Симуляция маршрутизации тикетов
- Unit tests + simulation tests

### Phase 2 — Contextual TS
- Logistic TS (binary reward + continuous features)
- Linear TS (continuous reward + features)
- Auto model selection по типу reward + context
- Расширение симуляции с признаками

### Phase 3 — API + Async + Persistence
- FastAPI endpoints (decide, feedback, admin)
- Async feedback через session_id + timeout
- SQLAlchemy + SQLite state store
- Integration tests

### Phase 4 — Advanced Features
- Forgetting/Decay (γ-decay для α,β)
- Shadow/A-B режим
- Arm lifecycle (add/remove/deactivate)
- Composite objective (multi-reward)
- Distribution collapse detection
- Batch updates
- State versioning и rollback

## Технологический стек
- Python 3.11+
- numpy, scipy — статистика и сэмплинг
- scikit-learn — LogisticRegression для Logistic TS (Phase 2)
- pydantic v2 — конфигурация и валидация
- pyyaml — загрузка конфигов
- pytest, hypothesis — тесты
- matplotlib, pandas — визуализация симуляции
- ruff — линтинг
- mypy — типизация

## Ключевые дизайн-решения

1. **Feedback синхронный в Phase 1**: в симуляции результат известен сразу.
   Async feedback — только в Phase 3.

2. **Session_id хранится в InMemoryStore**: словарь session_id → {arm_id, context, timestamp}.
   В Phase 3 заменяется на БД.

3. **Deterministic mode**: engine.decide(seed=42) даёт одинаковый результат.
   Обязательно для юнит-тестов.

4. **Интерфейс BaseModel одинаковый для всех моделей**: 
   sample(), update(), get_state(), load_state().
   Phase 2 добавляет новые реализации без изменения Engine.

5. **Context key для Beta categorical**: 
   sorted(context.items()) → "category_billing__tier_enterprise"
   Детерминированный, не зависит от порядка ключей.

6. **Reward нормализация**: reward передаётся напрямую в update().
   Если reward=1.0 → α+=1. Если reward=-0.5 → β+=0.5.
   Модель не знает про "сигналы" — агрегация происходит в FeedbackAggregator.