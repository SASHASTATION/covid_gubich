# =========================================================
# ✅ НАСТРОЙКА ОКРУЖЕНИЯ И ИМПОРТЫ
# - PYTENSOR_FLAGS: float64 → численная стабильность; optimizer=fast_run → быстрее граф
# - JAX_PLATFORMS: можно указать "cpu" или "metal" (на Apple) для ускорения через JAX
# =========================================================
import warnings
import os
os.environ.setdefault("PYTENSOR_FLAGS", "floatX=float64,optimizer=fast_run")
# os.environ.setdefault("JAX_PLATFORMS", "cpu")  # включайте при наличии JAX/JAX-Metal

import pymc as pm
try:
    # sample_numpyro_nuts — реализация NUTS через NumPyro/JAX: обычно быстрее, чем стандартный PyMC
    from pymc.sampling.jax import sample_numpyro_nuts
    USE_JAX = True
except Exception:
    sample_numpyro_nuts = None
    USE_JAX = False

warnings.simplefilter(action="ignore", category=FutureWarning)

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import pytensor
from pytensor.tensor.conv import conv2d
import pytensor.tensor as pt
from pytensor.scan import scan
from pytensor import config as ptconfig

# NP_FLOATX — тип чисел NumPy, согласованный с floatX PyTensor.
# Если floatX=float64 (как задано выше), то используем np.float64 для избежания даункаста.
NP_FLOATX = np.float64 if ptconfig.floatX == "float64" else np.float32


# =========================================================
# ✅ РАСПРЕДЕЛЕНИЕ ЗАДЕРЖКИ: Инкубация + задержка подтверждения
# - Конструирует дискретное PMF (вероятности по дням) через логнормаль.
# - INCUBATION_DAYS = фиксированный сдвиг (инкубационный период),
#   т.е. первые дни задержки «пустые» (нулевые).
# =========================================================
def get_delay_distribution():
    INCUBATION_DAYS = 5  # фиксируем 5 дней инкубации (сдвиг вправо)
    mean_si = 4.7        # средняя задержка «симптомы→подтверждение»
    std_si = 2.9         # СКО задержки
    # Перевод параметров среднего/СКО в (mu, sigma) логнормали:
    mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))
    sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))
    dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)

    max_delay = 60  # разумный «хвост» распределения, чтобы почти вся масса уложилась
    # Дискретизация CDF → PMF по дням: берём разности CDF
    p_delay = dist.cdf(np.arange(0, max_delay + 1))
    p_delay = np.diff(p_delay, prepend=0)
    p_delay /= p_delay.sum()  # нормируем, чтобы сумма = 1
    # Добавляем фиктивные нули для инкубационного периода
    p_delay = np.concatenate([np.zeros(INCUBATION_DAYS), p_delay])

    return p_delay.astype(NP_FLOATX)


# =========================================================
# ✅ ГЕНЕРАТИВНАЯ МОДЕЛЬ ЭПИДЕМИИ
# Суть:
# 1) Rt(t) — эффективное репродуктивное число (меняется каждые 3 дня, моделируется как GaussianRandomWalk на log-шкале)
# 2) infections(t) — рекурсия через вклад прошлых дней (serial interval), вычисляется через scan
# 3) positive(t) — ожидаемые подтверждённые случаи = infections ⨂ delay_distribution
# 4) Наблюдаемое — new_cases, шумим Negative Binomial (учёт сверхдисперсии)
# =========================================================
class GenerativeModel:
    def __init__(self, region: str, observed: pd.DataFrame, buffer_days=30, future_days=0):
        """
        Параметры:
        - region: метка региона (только для идентификации/логов).
        - observed: DataFrame с колонкой 'positive' (новые случаи в день).
        - buffer_days: «прогрев» до первой ненулевой точки (даём модели набрать инерцию заражений).
        - future_days: горизонт прогноза (сколько дней вперёд добавляем в индекс).
        """
        # Определяем первую ненулевую дату, чтобы не кормить модель «сплошными нулями»
        first_index = observed['positive'].ne(0).argmax()
        observed = observed.iloc[first_index:]

        # full_index = [buffer_days назад; …; исторические; …; future_days вперёд]
        new_index = pd.date_range(
            start=observed.index[0] - pd.Timedelta(days=buffer_days),
            end=observed.index[-1] + pd.Timedelta(days=future_days),
            freq="D",
        )
        self.full_index = new_index

        # Историческая часть (без буфера и будущего)
        historical_index = pd.date_range(
            start=observed.index[0],
            end=observed.index[-1],
            freq="D",
        )
        # Приводим к непрерывным датам (на случай пропусков)
        observed = observed.reindex(historical_index)
        observed['positive'] = observed['positive'].fillna(0)

        # Сохраняем служебные величины
        self.region = region
        self.buffer_days = buffer_days
        self.future_days = future_days
        self.observed = observed
        self.len_historical = len(historical_index)  # длина ряда, по которому есть наблюдения
        self.len_full = len(new_index)               # длина всего индекса (с буфером и будущим)
        self.model = None
        self.trace = None
        self.n_steps = None

    # ===============================
    # Дискретный сериал-интервал (serial interval) — PMF
    # почему до 20 дней? масса распределения после 20 мала
    # ===============================
    def _get_generation_time_interval(self):
        mean_si = 4.7
        std_si = 2.9
        mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))
        sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))
        dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)

        g_range = np.arange(0, 20)  # верхняя граница — баланс точности и скорости
        gt = np.diff(dist.cdf(g_range), prepend=0)  # CDF→PMF
        gt /= gt.sum()  # нормировка
        return gt.astype(NP_FLOATX)

    # Подготовка матрицы для быстрой свёртки в scan (оптимизация: избегаем слайсов в цикле PyTensor)
    def _get_convolution_ready_gt(self, len_full):
        gt = self._get_generation_time_interval()
        # Матрица размера (len_full-1, len_full): в строке t лежит gt, выровненный под момент t
        convolution_ready_gt = np.zeros((len_full - 1, len_full), dtype=NP_FLOATX)
        for t in range(1, len_full):
            begin = max(0, t - len(gt) + 1)
            # Берём gt[1: ...] и переворачиваем → соответствует сумме y * gt в scan_fn
            slice_update = gt[1: t - begin + 1][::-1]
            convolution_ready_gt[t - 1, begin: begin + len(slice_update)] = slice_update
        # shared → чтобы граф не копировал массив на каждом шаге
        return pytensor.shared(convolution_ready_gt)

    def build(self):
        """
        Сборка графа PyMC:
        - координаты (coords) → для удобных именованных размерностей и вывода
        - Rt(t) на лог-шкале как GRW с шагом 3 дня (сглаживание траектории)
        - рекурсивное вычисление infections через scan
        - свёртка infections с задержками (conv2d) → expected positives
        - likelihood: NegativeBinomial с параметрами mu, alpha
        """
        p_delay_np = get_delay_distribution().astype(NP_FLOATX)
        # Здесь nonzero_days = все True, т.к. мы уже обрезали ряд; оставлено как «маска» для гибкости
        nonzero_days = np.full(self.len_historical, True)
        convolution_ready_gt = self._get_convolution_ready_gt(self.len_full)

        # coords — метки осей для ArviZ/вытаскивания трассы
        coords = {
            "date": self.full_index.values,
            "nonzero_date": self.full_index[:self.len_historical][nonzero_days].values,
        }

        step = 3  # шаг «кубиков» Rt (каждые 3 дня Rt может плавно меняться)
        n_steps = int(np.ceil(self.len_full / step))
        self.n_steps = n_steps

        with pm.Model(coords=coords) as self.model:
            # ---------- Rt-блок ----------
            # GaussianRandomWalk на log(Rt): обеспечивает плавность траектории Rt
            # sigma=0.035 — дисперсия шага (чем больше, тем «нервнее» Rt)
            log_r_coarse = pm.GaussianRandomWalk(
                "log_r_coarse",
                sigma=0.035,
                shape=n_steps,
                init_dist=pm.Normal.dist(0, 1),  # начальное распределение для первого узла GRW
            )
            # Раскладываем «кубики» по дням: repeat по 3 дня, отрезаем до len_full
            log_r_t = pm.Deterministic(
                "log_r_t",
                pt.repeat(log_r_coarse, step)[:self.len_full],
                dims="date",
            )
            # Rt = exp(log Rt) → Rt > 0 по определению
            r_t = pm.Deterministic("r_t", pt.exp(log_r_t), dims="date")

            # ---------- infections-рекурсия ----------
            # seed — стартовая «инфекционная масса» (HalfNormal):
            # крупная сигма (1e4), чтобы не зажимать начальный уровень
            seed = pm.HalfNormal("seed", sigma=1e4)
            # y0 — вектор инфекций по всем дням; заполняем нулями и кладём seed в день 0
            y0 = pt.zeros(self.len_full, dtype=ptconfig.floatX)
            y0 = pt.set_subtensor(y0[0], seed)

            # scan_fn описывает обновление infections:
            # y[t] = Rt[t] * sum(y * gt_t), где gt_t — соответствующая строка готовой матрицы gt
            def scan_fn(t, gt, y, r_t):
                # y — аккумулятор (вектор длины len_full)
                # Здесь pt.sum(y * gt) — вклад предыдущих дней с весами сериал-интервала
                return pt.set_subtensor(y[t], r_t[t] * pt.sum(y * gt))

            # scan выполняет итеративную свёртку:
            # sequences: t (1..T), соответствующая строка gt
            # outputs_info: начальное состояние (y0)
            # non_sequences: r_t — внешняя последовательность Rt(t)
            outputs, _ = scan(
                fn=scan_fn,
                sequences=[pt.arange(1, self.len_full), convolution_ready_gt],
                outputs_info=y0,
                non_sequences=r_t,
                n_steps=self.len_full - 1,
            )
            # outputs[-1] — финальный вектор y со значениями infections по всем дням
            infections = pm.Deterministic("infections", outputs[-1], dims="date")

            # ---------- свёртка задержек ----------
            # conv2d ожидает 4D тензоры (N,C,H,W) — упаковываем инфекции и PMF задержек
            infections_4d = pt.reshape(pt.cast(infections, ptconfig.floatX), (1, 1, 1, self.len_full))
            p_delay_4d = pt.reshape(pt.as_tensor_variable(p_delay_np), (1, 1, 1, p_delay_np.shape[0]))

            # border_mode="full" — берём полную свёртку, затем обрезаем до len_full
            test_adjusted_positive = pm.Deterministic(
                "test_adjusted_positive",
                conv2d(infections_4d, p_delay_4d, border_mode="full")[0, 0, 0, : self.len_full],
                dims="date"
            )

            # В этой постановке exposure=1 (нет данных о тестах/детекции → считаем константу)
            exposure = pm.Deterministic("exposure", pt.ones(self.len_full), dims="date")
            positive = pm.Deterministic("positive", exposure * test_adjusted_positive, dims="date")

            # ---------- likelihood ----------
            # Negative Binomial (mu, alpha):
            # - mu = математическое ожидание
            # - alpha = параметр дисперсии (чем больше, тем ближе к Poisson; чем меньше — сильнее overdispersion)
            alpha = pm.Gamma("alpha", mu=4, sigma=2)  # информативный слабый приор на дисперсию
            mu_hist = positive[:self.len_historical][nonzero_days] + 1e-10  # +1e-10 чтобы избежать нулей в mu

            pm.NegativeBinomial(
                "nonzero_positive",
                mu=mu_hist,
                alpha=alpha,
                observed=self.observed['positive'][nonzero_days].values,
                dims="nonzero_date",
            )

        return self.model

    def sample(self, draws=10, tune=10, chains=4, target_accept=0.95, engine="jax"):
        """
        Семплирование постериора.
        Параметры MCMC:
        - draws: число сохранившихся выборок на цепь после прогрева (tune).
                 Итоговых семплов будет draws * chains.
        - tune: длина прогрева/адаптации (burn-in). Эти шаги не сохраняются в итоговую трассу.
        - chains: число независимых цепей MCMC. 4 — стандарт для диагностики сходимости (r_hat).
        - target_accept: целевой уровень приёма шагов NUTS (0.8–0.95 обычно).
                         Чем выше, тем меньше шаг интегратора → стабильнее, но медленнее.
        - engine: "jax" → sample_numpyro_nuts (если доступен), иначе классический pm.sample (PyMC).
        Важно: random_seed фиксируем для воспроизводимости.
        """
        if self.model is None:
            self.build()

        # start — начальные точки для некоторых параметров;
        # помогает избежать плохой инициализации (особенно при небольшом количестве данных).
        start = {"log_r_coarse": np.full(self.n_steps, 0.0), "seed": 100.0, "alpha": 4.0}

        if engine == "jax" and USE_JAX:
            # chain_method="vectorized" — параллелим цепи векторно (быстрее на JAX бекенде)
            self.trace = sample_numpyro_nuts(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                chain_method="vectorized",
                model=self.model,
                progressbar=True,
                random_seed=42,
            )
            return self
        else:
            with self.model:
                # init="advi+adapt_diag" — быстрая вариационная инициализация + адаптивная диагональная ковариация
                # cores=1 — избегаем проблем многопроцессности в некоторых средах (но можно увеличить)
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=1,
                    target_accept=target_accept,
                    init="advi+adapt_diag",
                    start=start,
                    compute_convergence_checks=True,  # r_hat, ESS и т.д.
                    progressbar=True,
                    random_seed=42,
                )
            return self


# =========================================================
# ✅ ОБРАБОТКА ОДНОЙ СТРАНЫ
# - train_end: дата, до которой обучаем (включительно)
# - future_days: запас дат вперёд в индексе модели (для прогноза/экстраполяции)
# Неочевидные шаги:
# - стартуем, когда new_cases > 100 → уменьшаем влияние раннего шумного периода
# - posterior predictive: берём 'positive' (ожидаемые случаи) и усредняем по (chain, draw)
# =========================================================
def process_country(country, owid_df, train_end='2020-12-01', future_days=13):
    country_data = owid_df[owid_df['location'] == country].set_index('date')

    # Стартовая дата — первая, где новые случаи > 100 (стабильнее динамика Rt)
    start_date = country_data[country_data['new_cases'] > 100].index.min()
    if pd.isnull(start_date):
        print(f"No data >100 for {country}")
        return None

    # Отрезаем ряд до train_end — это обучающая часть
    country_data = country_data.loc[start_date:train_end]
    observed = pd.DataFrame({'positive': country_data['new_cases'].fillna(0)})

    # Инициализация модели (buffer_days внутри конструктора добавит «прогрев» влево)
    gm = GenerativeModel(country, observed, future_days=future_days)
    gm.sample()  # семплируем постериор Rt, infections, и пр.

    # Posterior Predictive:
    # extend_inferencedata=True → добавляет раздел posterior_predictive в уже существующий InferenceData
    with gm.model:
        inference_data = pm.sample_posterior_predictive(
            gm.trace,
            var_names=["positive"],  # берём предиктивное распределение ожидаемых случаев
            progressbar=False,
            extend_inferencedata=True
        )

    # Сводка Rt по датам: mean + 80% HDI (hdi_prob=0.8 → центральный интервал неопределённости)
    rt_summary = az.summary(inference_data.posterior['r_t'], hdi_prob=0.8)

    # Средняя предиктивная траектория: усредняем по всем сэмплам и цепям
    predicted_full = inference_data.posterior['positive'].mean(dim=['chain', 'draw'])

    # Фактические данные «будущего» для проверки прогноза (фиксированный диапазон)
    real_future = owid_df[
        (owid_df['location'] == country)
        & (owid_df['date'] > train_end)
        & (owid_df['date'] <= '2020-12-14')
    ][['date', 'new_cases']]

    return rt_summary, predicted_full, real_future


# =========================================================
# ✅ MAIN: Загрузка данных и запуск по списку стран
# Неочевидные моменты:
# - источник OWID/JHU: CSV напрямую с GitHub raw
# - parse_dates=['date'] — сразу распознаём даты
# - цикл по странам: для каждой строим модель отдельно (независимые прогоны)
# =========================================================
if __name__ == "__main__":
    owid_url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv"
    owid_response = requests.get(owid_url)
    owid_df = pd.read_csv(StringIO(owid_response.text), parse_dates=['date'])

    countries = ['Russia', 'Italy', 'Germany', 'France']
    for country in countries:
        print(f"Processing {country}...")
        process_country(country, owid_df)
