Ты создаёшь Phase 1 универсального TS-модуля.
Прочитай PROJECT_CONTEXT.md перед началом — там полный контекст проекта,
архитектура, дизайн-решения и интерфейсы.

═══════════════════════════════════════════════════════
ЗАДАЧА PHASE 1
═══════════════════════════════════════════════════════

Реализовать ядро модуля (Beta TS) + симуляцию маршрутизации тикетов.
После Phase 1 должно работать: python examples/ticket_routing/simulate.py
и pytest tests/ — все тесты зелёные.

═══════════════════════════════════════════════════════
ШАГ 1 — СТРУКТУРА ПРОЕКТА
═══════════════════════════════════════════════════════

Создай следующую структуру файлов:

ts_module/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── engine.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── beta.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── memory.py
│   ├── objective.py
│   ├── constraints.py
│   └── feedback.py
├── config/
│   ├── __init__.py
│   ├── schema.py
│   ├── validator.py
│   └── loader.py
examples/
└── ticket_routing/
    ├── config.yaml
    ├── simulate.py
    ├── generator.py
    └── visualize.py
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_beta_model.py
│   ├── test_objective.py
│   ├── test_constraints.py
│   ├── test_feedback.py
│   └── test_config.py
└── simulation/
    ├── __init__.py
    └── test_ticket_routing.py

═══════════════════════════════════════════════════════
ШАГ 2 — КОНФИГУРАЦИЯ (config/)
═══════════════════════════════════════════════════════

config/schema.py — Pydantic v2 модели для всей конфигурации модуля.

Реализуй следующие классы:

  ContextMode(str, Enum): none | categorical | features
  ModelType(str, Enum): auto | beta | logistic | linear
  RewardType(str, Enum): binary | continuous | composite
  ObjectiveType(str, Enum):
      max_probability | max_expected_revenue | max_roi |
      min_cost_per_success | custom

  ArmConfig(BaseModel):
      id: str
      name: str
      metadata: dict = {}

  ContextFeatureConfig(BaseModel):
      name: str
      type: str  # "categorical" | "continuous"

  ContextConfig(BaseModel):
      mode: ContextMode = none
      features: list[ContextFeatureConfig] = []

  SignalConfig(BaseModel):
      name: str
      weight: float = 1.0
      timeout_hours: int = 72
      default_on_timeout: float = 0.0

  RewardConfig(BaseModel):
      type: RewardType = binary
      signals: list[SignalConfig]
      aggregation: str = "weighted_sum"

  ObjectiveConfig(BaseModel):
      type: ObjectiveType = max_probability
      formula: str | None = None

  ConstraintConfig(BaseModel):
      type: str  # "capacity" | "min_traffic" | "eligibility"
      arm_id: str | None = None
      value: float | None = None
      arm_field: str | None = None
      condition: str | None = None

  BetaHyperparams(BaseModel):
      alpha_init: float = 1.0
      beta_init: float = 1.0
      forgetting_rate: float = 0.97
      min_exploration: float = 0.03
      cold_start_pulls: int = 30

  HyperparamsConfig(BaseModel):
      model_type: ModelType = auto
      beta: BetaHyperparams = BetaHyperparams()
      update_mode: str = "realtime"

  ModuleConfig(BaseModel):
      id: str
      name: str
      arms: list[ArmConfig]
      context: ContextConfig = ContextConfig()
      reward: RewardConfig
      objective: ObjectiveConfig = ObjectiveConfig()
      constraints: list[ConstraintConfig] = []
      hyperparams: HyperparamsConfig = HyperparamsConfig()

      @model_validator(mode='after')
      def auto_select_model_type(self):
          # Если model_type == auto:
          # binary + (none|categorical) → beta
          # binary + features → logistic
          # continuous + any → linear
          # Записать результат обратно в hyperparams.model_type
          ...

      @model_validator(mode='after')
      def validate_arm_ids_unique(self):
          # Если есть дубли — raise ValueError
          ...

      @model_validator(mode='after')
      def validate_custom_objective_has_formula(self):
          # Если type==custom и formula==None — raise ValueError
          ...

config/loader.py — загрузка конфига из YAML-файла или dict:
  load_from_yaml(path: str | Path) -> ModuleConfig
  load_from_dict(data: dict) -> ModuleConfig

═══════════════════════════════════════════════════════
ШАГ 3 — МОДЕЛИ (core/models/)
═══════════════════════════════════════════════════════

core/models/base.py — абстрактный базовый класс:

  class BaseModel(ABC):
      @abstractmethod
      def sample(self, arm_id: str, context: dict,
                 seed: int | None = None) -> float: ...

      @abstractmethod
      def update(self, arm_id: str, reward: float,
                 context: dict) -> None: ...

      @abstractmethod
      def get_distribution(self, arm_id: str,
                           context: dict) -> dict: ...
      # Возвращает {"alpha": float, "beta": float, "estimated_p": float}
      # estimated_p = alpha / (alpha + beta)

      @abstractmethod
      def get_all_distributions(self) -> dict: ...
      # Возвращает весь стейт: {arm_id: {context_key: {alpha, beta, estimated_p}}}

      @abstractmethod
      def get_state(self) -> dict: ...

      @abstractmethod
      def load_state(self, state: dict) -> None: ...

core/models/beta.py — реализация Beta TS:

  class BetaModel(BaseModel):
      Инициализация:
        - arms: list[ArmConfig]
        - context_config: ContextConfig
        - hyperparams: BetaHyperparams
        - Внутренний стор: dict[(arm_id, context_key), (alpha, beta)]

      _make_context_key(self, context: dict) -> str:
        - Если context_mode == none: вернуть "global"
        - Если context_mode == categorical:
          Взять только поля из context_config.features
          Отсортировать по имени: sorted(items)
          Собрать строку: "category_billing__tier_enterprise"
        - Ключ должен быть детерминированным (не зависит от порядка dict)

      _get_or_init(self, arm_id: str, context_key: str) -> tuple[float, float]:
        - Вернуть существующие (alpha, beta) или инициализировать
          из hyperparams.alpha_init, hyperparams.beta_init

      sample(self, arm_id: str, context: dict,
             seed: int | None = None) -> float:
        - Получить context_key
        - Получить или инициализировать (alpha, beta)
        - rng = np.random.default_rng(seed)
        - Вернуть float(rng.beta(alpha, beta))

      update(self, arm_id: str, reward: float, context: dict) -> None:
        - Получить context_key
        - Получить (alpha, beta)
        - Если reward > 0: alpha += reward
        - Если reward < 0: beta += abs(reward)
        - Если reward == 0: beta += 1.0
        - Сохранить новые значения

      get_distribution(self, arm_id: str, context: dict) -> dict:
        - Вернуть {"alpha": α, "beta": β, "estimated_p": α/(α+β)}

      get_all_distributions(self) -> dict:
        - Вернуть {arm_id: {context_key: {alpha, beta, estimated_p}}}

      get_state(self) -> dict:
        - Сериализовать весь стор в JSON-совместимый dict

      load_state(self, state: dict) -> None:
        - Восстановить стор из dict

═══════════════════════════════════════════════════════
ШАГ 4 — STATE STORE (core/state/)
═══════════════════════════════════════════════════════

core/state/base.py — абстрактный стор для хранения сессий:

  class BaseStateStore(ABC):
      @abstractmethod
      def save_session(self, session_id: str,
                       data: SessionData) -> None: ...

      @abstractmethod
      def get_session(self, session_id: str) -> SessionData | None: ...

      @abstractmethod
      def delete_session(self, session_id: str) -> None: ...

  @dataclass
  class SessionData:
      session_id: str
      arm_id: str
      context: dict
      timestamp: datetime
      signals_received: list[dict] = field(default_factory=list)
      is_finalized: bool = False

core/state/memory.py — in-memory реализация:
  class InMemoryStateStore(BaseStateStore):
      Хранит сессии в словаре {session_id: SessionData}
      Реализует все методы базового класса

═══════════════════════════════════════════════════════
ШАГ 5 — OBJECTIVE + CONSTRAINTS + FEEDBACK
═══════════════════════════════════════════════════════

core/objective.py:

  class ObjectiveFunction:
      def __init__(self, config: ObjectiveConfig): ...

      def score(self, arm_id: str, theta: float,
                arm_metadata: dict, context: dict) -> float:
        # max_probability: вернуть theta
        # max_expected_revenue: theta * arm_metadata.get("price", 1.0)
        # max_roi: (theta * arm_metadata.get("value",1) -
        #           arm_metadata.get("cost",0)) / max(arm_metadata.get("cost",1), 0.001)
        # min_cost_per_success: -arm_metadata.get("cost",1) / max(theta, 0.001)
        # custom: eval формулы с переменными theta, arm (=arm_metadata), context

      def select(self, scored_arms: dict[str, float]) -> str:
        # Вернуть arm_id с максимальным score
        # При равенстве — случайный из максимальных

  ВАЖНО для custom formula:
    Безопасный eval через ограниченный namespace:
    {"theta": theta, "arm": arm_metadata, "context": context, "__builtins__": {}}
    Поймать исключения и поднять ValueError с понятным сообщением

core/constraints.py:

  class ConstraintEngine:
      def __init__(self, constraints: list[ConstraintConfig],
                   arms: list[ArmConfig]): ...

      def filter_eligible(
          self,
          arms: list[str],
          context: dict,
          current_loads: dict[str, int] | None = None
      ) -> list[str]:
        # capacity: исключить arm если current_loads[arm_id] >= arm.metadata[field]
        # eligibility: исключить arm если условие не выполнено
        # Всегда оставить хотя бы одну руку (если все исключены — вернуть все)

      def apply_exploration_floor(
          self,
          arm_scores: dict[str, float],
          min_exploration: float
      ) -> dict[str, float]:
        # Гарантировать что ни одна рука не получит меньше min_exploration трафика
        # Реализация: если score слишком мал — поднять до
        # min_exploration * max_score / (1 - min_exploration * n_arms)
        # Нормализовать итоговые scores

core/feedback.py:

  class FeedbackAggregator:
      def __init__(self, reward_config: RewardConfig): ...

      def aggregate(self, signals: list[dict]) -> float:
        # signals = [{"name": "fcr", "value": 1.0}, ...]
        # weighted_sum: sum(signal["value"] * weight для каждого сигнала)
        # first_positive: вернуть первый сигнал с value > 0
        # max: вернуть максимальное значение сигнала

        # Если сигнал не найден в reward_config.signals — игнорировать
        # Если ни одного сигнала — вернуть 0.0

═══════════════════════════════════════════════════════
ШАГ 6 — ENGINE (core/engine.py)
═══════════════════════════════════════════════════════

Dataclasses для результатов:

  @dataclass
  class DecisionResult:
      session_id: str
      recommended_arm: str
      confidence: float
      arm_scores: dict[str, float]
      model_state_snapshot: dict

  @dataclass
  class UpdateResult:
      session_id: str
      arm_updated: str
      reward_applied: float
      new_distribution: dict

Главный класс:

  class TSEngine:
      def __init__(self, config: ModuleConfig):
          # Создать модель (в Phase 1 только BetaModel)
          # Создать InMemoryStateStore
          # Создать ObjectiveFunction
          # Создать ConstraintEngine
          # Создать FeedbackAggregator

      def decide(
          self,
          context: dict,
          eligible_arms: list[str] | None = None,
          session_id: str | None = None,
          seed: int | None = None,
          current_loads: dict[str, int] | None = None
      ) -> DecisionResult:
          # 1. Если eligible_arms is None → взять все arms из конфига
          # 2. Применить ConstraintEngine.filter_eligible()
          # 3. Если после фильтрации 0 рук → использовать все arms
          # 4. Сэмплировать θk для каждой руки: model.sample(arm_id, context, seed)
          # 5. Вычислить score через ObjectiveFunction.score()
          # 6. Применить exploration floor
          # 7. Выбрать лучшую руку: ObjectiveFunction.select()
          # 8. confidence = best_score - second_best_score (0.0 если одна рука)
          # 9. Сохранить сессию в StateStore
          # 10. Сделать snapshot модели для аудита
          # 11. Вернуть DecisionResult

      def feedback(
          self,
          session_id: str,
          signals: list[dict]
      ) -> UpdateResult:
          # 1. Получить сессию из StateStore по session_id
          # 2. Если не найдена → raise ValueError
          # 3. Агрегировать сигналы через FeedbackAggregator
          # 4. Обновить модель: model.update(arm_id, reward, context)
          # 5. Пометить сессию как finalized
          # 6. Вернуть UpdateResult с новым распределением

      def get_arm_state(
          self, arm_id: str, context: dict = {}
      ) -> dict:
          return model.get_distribution(arm_id, context)

      def get_all_states(self) -> dict:
          return model.get_all_distributions()

      @classmethod
      def from_yaml(cls, path: str) -> "TSEngine":
          config = load_from_yaml(path)
          return cls(config)

═══════════════════════════════════════════════════════
ШАГ 7 — СИМУЛЯЦИЯ (examples/ticket_routing/)
═══════════════════════════════════════════════════════

examples/ticket_routing/config.yaml:

  id: ticket_routing_v1
  name: "Маршрутизация тикетов к агентам"
  arms:
    - id: agent_maria
      name: "Мария С."
      metadata: {capacity: 8, cost_per_hour: 150}
    - id: agent_ivan
      name: "Иван К."
      metadata: {capacity: 8, cost_per_hour: 150}
    - id: agent_alexey
      name: "Алексей Д."
      metadata: {capacity: 8, cost_per_hour: 180}
    - id: agent_olga
      name: "Ольга Р."
      metadata: {capacity: 8, cost_per_hour: 150}
    - id: agent_dmitry
      name: "Дмитрий В."
      metadata: {capacity: 8, cost_per_hour: 120}
  context:
    mode: categorical
    features:
      - {name: ticket_category, type: categorical}
      - {name: client_tier, type: categorical}
  reward:
    type: binary
    signals:
      - {name: fcr,    weight: 1.0,  timeout_hours: 72, default_on_timeout: 0.0}
      - {name: reopen, weight: -0.5, timeout_hours: 72, default_on_timeout: 0.0}
      - {name: csat_high, weight: 0.3, timeout_hours: 96, default_on_timeout: 0.0}
    aggregation: weighted_sum
  objective:
    type: max_probability
  constraints:
    - {type: min_traffic, value: 0.03}
  hyperparams:
    model_type: auto
    beta:
      alpha_init: 1.0
      beta_init: 1.0
      min_exploration: 0.03
      cold_start_pulls: 30
    update_mode: realtime

examples/ticket_routing/generator.py:

  TRUE_FCR = {
      "agent_maria":  {"billing":0.72,"tech":0.68,"complaint":0.91,"onboard":0.70},
      "agent_ivan":   {"billing":0.81,"tech":0.74,"complaint":0.61,"onboard":0.79},
      "agent_alexey": {"billing":0.65,"tech":0.88,"complaint":0.59,"onboard":0.82},
      "agent_olga":   {"billing":0.78,"tech":0.71,"complaint":0.74,"onboard":0.93},
      "agent_dmitry": {"billing":0.70,"tech":0.65,"complaint":0.67,"onboard":0.68},
  }
  ORACLE = {
      "billing":   ("agent_ivan",   0.81),
      "tech":      ("agent_alexey", 0.88),
      "complaint": ("agent_maria",  0.91),
      "onboard":   ("agent_olga",   0.93),
  }
  TICKET_DIST = {
      "billing":0.35, "tech":0.28, "complaint":0.22, "onboard":0.15
  }
  CLIENT_TIERS = ["smb","mid","enterprise"]
  TIER_WEIGHTS = [0.5, 0.35, 0.15]

  def generate_ticket(rng: np.random.Generator) -> dict:
      # Вернуть {"ticket_category": ..., "client_tier": ...}
      # Случайная категория по TICKET_DIST, тир по TIER_WEIGHTS

  def simulate_fcr(agent_id: str, ticket: dict, rng: np.random.Generator) -> float:
      # Bernoulli(TRUE_FCR[agent_id][ticket["ticket_category"]])
      true_p = TRUE_FCR[agent_id][ticket["ticket_category"]]
      return 1.0 if rng.random() < true_p else 0.0

examples/ticket_routing/simulate.py:

  Принимать аргументы:
    --tickets (default: 1500)
    --seed (default: 42)
    --show-plots (flag)
    --save-results (flag, сохранить CSV в results/)

  Запустить три стратегии на одинаковых данных:
    1. round_robin(tickets): случайное равномерное назначение
    2. ts_strategy(tickets): использует TSEngine
    3. oracle_strategy(tickets): всегда назначает лучшего агента

  Для каждой стратегии собирать:
    - fcr_per_ticket: список FCR по каждому тикету (0 или 1)
    - cumulative_fcr: накопленный средний FCR
    - agent_assignments: сколько тикетов досталось каждому агенту
    - regret_vs_oracle: накопленный regret = oracle_fcr - strategy_fcr

  Вывод в консоль (использовать rich.table):
    Ticket 500:  TS FCR=0.798 | RR FCR=0.731 | Oracle FCR=0.882
    Ticket 1000: TS FCR=0.841 | RR FCR=0.733 | Oracle FCR=0.882
    Ticket 1500: TS FCR=0.858 | RR FCR=0.731 | Oracle FCR=0.882

    Agent specialization after 1500 tickets (TS):
    | Agent       | billing | tech | complaint | onboard | Total |
    | agent_maria |    8%   |  9%  |   71%     |  12%    |  200  |
    ...

examples/ticket_routing/visualize.py:

  def plot_cumulative_fcr(results: dict, save_path=None):
      # Три линии: TS (оранжевая), Round-robin (серая), Oracle (зелёная пунктир)
      # X: номер тикета, Y: накопленный FCR

  def plot_regret(results: dict, save_path=None):
      # Накопленный regret TS vs Round-robin
      # Должен расти медленнее у TS

  def plot_agent_traffic(ts_assignments: dict, save_path=None):
      # Stacked bar chart: трафик по агентам для каждой категории

═══════════════════════════════════════════════════════
ШАГ 8 — ТЕСТЫ
═══════════════════════════════════════════════════════

tests/unit/test_beta_model.py (минимум 15 тестов):

  test_sample_in_unit_interval()
      Для всех arms, sample всегда в [0.0, 1.0]

  test_success_increments_alpha()
      update(reward=1.0) → alpha увеличился на 1, beta не изменился

  test_failure_increments_beta()
      update(reward=0.0) → beta увеличился на 1, alpha не изменился

  test_negative_reward_increments_beta_by_absolute_value()
      update(reward=-0.5) → beta увеличился на 0.5

  test_partial_reward_increments_alpha_by_value()
      update(reward=0.3) → alpha увеличился на 0.3

  test_categorical_context_creates_independent_distributions()
      update billing context → не влияет на complaint context

  test_global_context_key_when_mode_none()
      При context_mode=none все обновления идут в "global"

  test_context_key_is_deterministic_regardless_of_dict_order()
      context={"b":"2","a":"1"} и context={"a":"1","b":"2"} → одинаковый ключ

  test_convergence_to_true_probability()
      200 обновлений с true_p=0.75 → estimated_p близко к 0.75 (±0.05)

  test_deterministic_sampling_with_seed()
      sample(seed=42) == sample(seed=42)

  test_different_seeds_give_different_samples()
      sample(seed=42) != sample(seed=99) (почти всегда)

  test_initial_distribution_from_hyperparams()
      alpha_init=2, beta_init=5 → начальные alpha=2, beta=5

  test_get_all_distributions_returns_all_arms()
      После обновлений нескольких arms — get_all_distributions содержит все

  test_state_serialization_roundtrip()
      get_state() → load_state() → те же alpha/beta

  test_unknown_arm_raises_value_error()
      sample("nonexistent_arm", {}) → ValueError

tests/unit/test_objective.py (8 тестов):

  test_max_probability_returns_theta()
  test_max_revenue_multiplies_by_price()
  test_max_roi_formula()
  test_min_cost_per_success()
  test_select_returns_highest_score_arm()
  test_select_handles_single_arm()
  test_custom_formula_with_theta_and_arm_metadata()
  test_invalid_custom_formula_raises_value_error()

tests/unit/test_constraints.py (6 тестов):

  test_capacity_excludes_overloaded_arm()
  test_capacity_keeps_arm_under_limit()
  test_min_traffic_floor_applied_to_all_arms()
  test_all_arms_excluded_returns_all_arms()  # защита от пустого списка
  test_exploration_floor_normalizes_scores()
  test_no_constraints_returns_all_arms()

tests/unit/test_feedback.py (6 тестов):

  test_single_signal_aggregation()
  test_weighted_sum_multiple_signals()
  test_negative_weight_signal()
  test_unknown_signal_name_ignored()
  test_empty_signals_returns_zero()
  test_first_positive_aggregation_mode()

tests/unit/test_config.py (8 тестов):

  test_auto_model_type_binary_categorical_resolves_to_beta()
  test_auto_model_type_binary_features_resolves_to_logistic()
  test_auto_model_type_continuous_resolves_to_linear()
  test_duplicate_arm_ids_raise_validation_error()
  test_custom_objective_without_formula_raises_error()
  test_load_from_yaml_returns_module_config()
  test_invalid_yaml_raises_error()
  test_missing_required_fields_raise_validation_error()

tests/simulation/test_ticket_routing.py (5 тестов):

  test_ts_fcr_beats_round_robin_after_500_tickets()
      После 500 тикетов средний FCR у TS > FCR у Round-robin на > 3%
      seed=42 для воспроизводимости
      Запустить 3 раза с разными seeds и проверить что TS лучше в каждом

  test_ts_regret_grows_slower_than_round_robin()
      Наклон regret TS между тикетами 400-500 < наклон между 0-100

  test_ts_learns_complaint_specialist_within_300_tickets()
      После 300 тикетов: agent_maria получает >50% complaint-тикетов

  test_ts_learns_tech_specialist_within_300_tickets()
      После 300 тикетов: agent_alexey получает >50% tech-тикетов

  test_oracle_fcr_is_upper_bound()
      FCR Oracle >= FCR TS >= FCR Round-robin (всегда)

═══════════════════════════════════════════════════════
КРИТЕРИИ ГОТОВНОСТИ PHASE 1
═══════════════════════════════════════════════════════

1. pytest tests/ → все тесты зелёные, 0 failures
2. python examples/ticket_routing/simulate.py → запускается без ошибок,
   выводит таблицу метрик, TS FCR > Round-robin FCR
3. ruff check ts_module/ → 0 ошибок
4. mypy ts_module/ → 0 critical errors (ignore_missing_imports=true)
5. Весь публичный API аннотирован типами
6. Все классы и методы имеют docstrings

═══════════════════════════════════════════════════════
ВАЖНЫЕ ДИЗАЙН-ОГРАНИЧЕНИЯ
═══════════════════════════════════════════════════════

1. В Phase 1 НЕТ: FastAPI, SQLAlchemy, aiosqlite, async кода.
   Всё синхронно, всё в памяти.

2. Интерфейс BaseModel.sample() и .update() должен быть финальным —
   Phase 2 добавит LogisticModel и LinearModel без изменения Engine.

3. engine.decide(seed=42) должен давать одинаковый результат
   при одинаковом состоянии модели. Это обязательно для тестов.

4. При невалидной конфигурации — ранний raise с понятным сообщением,
   не молчаливый fallback.

5. Логировать каждое решение и обновление через стандартный logging
   (уровень DEBUG для сэмплов, INFO для решений и обновлений).