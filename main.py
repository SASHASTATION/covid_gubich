import warnings
import os
os.environ.setdefault("PYTENSOR_FLAGS", "floatX=float64,optimizer=fast_run")  # Set to float64 for numerical stability in computations
# Для Apple Silicon: если JAX установлен, раскомментируйте
# os.environ.setdefault("JAX_PLATFORMS", "cpu")  # Or "metal" if jax-metal is installed, to specify backend for JAX

import pymc as pm  # Import PyMC (version 5+) for Bayesian modeling; this is the main library for stochastic generative modeling
try:
    from pymc.sampling.jax import sample_numpyro_nuts  # Import NumPyro sampler from PyMC for faster sampling using JAX if available
    USE_JAX = True  # Flag to indicate if JAX sampler is available
except Exception:
    sample_numpyro_nuts = None  # Set to None if import fails
    USE_JAX = False  # Flag to indicate JAX sampler is not available

warnings.simplefilter(action="ignore", category=FutureWarning)  # Suppress future warnings to clean up output

import arviz as az  # For analyzing posterior distributions (e.g., r_hat, HDI, inference data)
import numpy as np  # For numerical operations and arrays
import pandas as pd  # For data handling (DataFrames)
from scipy import stats as sps  # For statistical distributions (e.g., lognorm)
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhancing plots (e.g., despine)
import requests  # For downloading data from URL
from io import StringIO  # For reading CSV from strings
import pytensor  # PyTensor for tensors and compilation to C for acceleration
from pytensor.tensor.conv import conv2d  # Convolution for modeling delays (reporting delay + incubation)
import pytensor.tensor as pt  # PyTensor tensor operations
from pytensor.scan import scan  # For looping in Theano/PyTensor graphs
from pytensor import config as ptconfig  # Access PyTensor configuration
NP_FLOATX = np.float64 if ptconfig.floatX == "float64" else np.float32  # Adapt dtype based on PyTensor config

# =========================================================
# РАСПРЕДЕЛЕНИЕ ЗАДЕРЖКИ: Инкубация + задержка подтверждения
# - моделируем логнормаль распределение → PMF задержек
# - добавляем фиксированные 5 дней инкубации
# =========================================================
def get_delay_distribution():
    """ Returns the distribution of delays (incubation period of 5 days + delay from symptoms to confirmation).
    This is an empirical distribution based on patient line list data.
    """
    INCUBATION_DAYS = 5  # Fixed incubation period from literature
    mean_si = 4.7  # Mean symptom to confirmation delay
    std_si = 2.9  # Standard deviation of symptom to confirmation delay
    mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))  # Parameter for lognormal distribution
    sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))  # Parameter for lognormal distribution
    dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)  # Lognormal distribution object
    max_delay = 60  # Maximum days for delay
    p_delay = dist.cdf(np.arange(0, max_delay + 1))  # Cumulative distribution function for discretization
    p_delay = np.diff(p_delay, prepend=0)  # Convert to probability mass function
    p_delay /= p_delay.sum()  # Normalize to ensure sums to 1
    p_delay = np.concatenate([np.zeros(INCUBATION_DAYS), p_delay])  # Add incubation zeros at the beginning
    return p_delay.astype(NP_FLOATX)  # Return as array with appropriate dtype

# =========================================================
# ГЕНЕРАТИВНАЯ МОДЕЛЬ
# Главные элементы:
# 1️⃣ Rt(t) — случайная величина, меняется каждые 3 дня → GRW
# 2️⃣ infections(t) — модель инфекций через свёртку со serial interval
# 3️⃣ positive(t) = infections ⨂ delay_distribution
#
# 👉 Наблюдаемая величина → new_cases ≈ Negative Binomial(μ, α)
# ▪ данные по случаям сверхдисперсные (variance > mean)
# ▪ Poisson бы давал неверный доверительный интервал
# =========================================================
class GenerativeModel:
    def __init__(self, region: str, observed: pd.DataFrame, buffer_days=30, future_days=0):
        # Increased buffer_days for better ramp up of infections
        """ Initialization of the model.
        - region: Country name (for logging).
        - observed: DataFrame with 'positive' (new_cases); 'total' not used since absent in JHU data.
        - buffer_days: Padding days at the beginning for infections before first cases.
        - future_days: Days for forecasting.
        """
        # ✅ Подготовка временных индексов и сохраняем параметры
        self.region = region  # Store region name
        self.buffer_days = buffer_days  # Store buffer days
        self.future_days = future_days  # Store future days
        first_index = observed['positive'].ne(0).argmax()  # Find first non-zero day (though start_date already >100)
        observed = observed.iloc[first_index:]  # Trim to first case
        new_index = pd.date_range(  # Extend index: buffer_days back + future_days forward
            start=observed.index[0] - pd.Timedelta(days=buffer_days),
            end=observed.index[-1] + pd.Timedelta(days=future_days),
            freq="D",
        )
        self.full_index = new_index  # Full index (historical + future)
        historical_index = pd.date_range(start=observed.index[0], end=observed.index[-1], freq="D")  # Historical index
        observed = observed.reindex(historical_index)  # Reindex historical data
        observed['positive'] = observed['positive'].fillna(0)  # Fill missing positive with 0
        self.observed = observed  # Store observed data
        self.len_historical = len(historical_index)  # Length of historical data
        self.len_full = len(new_index)  # Full length including future
        self.model = None  # PyMC model (built in build method)
        self.trace = None  # Sampling trace
        self.n_steps = None  # Number of steps for Rt changes

    # ===============================
    # Серийный интервал генераций
    # дискретный логнормаль PMF
    # ===============================
    def _get_generation_time_interval(self):
        """ Discrete distribution for generation interval (lognorm from literature).
        """
        mean_si = 4.7  # Mean generation interval
        std_si = 2.9  # Standard deviation
        mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))  # Lognorm parameter
        sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))  # Lognorm parameter
        dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)  # Lognormal distribution
        g_range = np.arange(0, 20)  # Up to 20 days
        gt = dist.cdf(g_range)  # CDF
        gt = np.diff(gt, prepend=0)  # PMF
        gt /= gt.sum()  # Normalize
        return gt.astype(NP_FLOATX)  # Return with dtype

    # Готовим сериал-интервал для ускоренной свёртки в scan()
    def _get_convolution_ready_gt(self, len_full):
        """ Prepare generation interval for fast convolution in scan (optimization).
        """
        gt = self._get_generation_time_interval()  # Get gt
        convolution_ready_gt = np.zeros((len_full - 1, len_full), dtype=NP_FLOATX)  # Initialize array
        for t in range(1, len_full):  # Fill reversed gt slices
            begin = max(0, t - len(gt) + 1)
            slice_update = gt[1: t - begin + 1][::-1]
            convolution_ready_gt[t - 1, begin: begin + len(slice_update)] = slice_update
        return pytensor.shared(convolution_ready_gt)  # Shared variable for PyTensor

    # =========================================================
    # СБОРКА PyMC5-МОДЕЛИ
    # - Rt ~ GRW (log-шкала → Rt = exp(log_r_t))
    # - infections через рекурсивную свёртку
    # - positive = infections * delay_PMF
    # - likelihood → NegativeBinomial
    # =========================================================
    def build(self):
        """ Build the PyMC model.
        """
        p_delay_np = get_delay_distribution().astype(NP_FLOATX)  # Get delay PMF
        # Since no 'total' in JHU, assume all historical days are observed
        nonzero_days = np.full(self.len_historical, True)  # All days after start are considered
        convolution_ready_gt = self._get_convolution_ready_gt(self.len_full)  # Prepare gt for convolution
        coords = {  # Coordinates for dims in model
            "date": self.full_index.values,
            "nonzero_date": self.full_index[:self.len_historical][nonzero_days].values,
        }
        step = 3  # Rt changes every 3 days
        n_steps = int(np.ceil(self.len_full / step))  # Number of coarse steps
        self.n_steps = n_steps  # Save for init
        with pm.Model(coords=coords) as self.model:  # Create PyMC model context
            log_r_coarse = pm.GaussianRandomWalk(  # Gaussian random walk for log Rt on coarse scale
                "log_r_coarse",
                sigma=0.035,  # Standard deviation of walk
                shape=n_steps,
                init_dist=pm.Normal.dist(0, 1),  # Initial distribution
            )
            #  seed ~ prior — стартовая инфекционная масса
            log_r_t = pm.Deterministic("log_r_t", pt.repeat(log_r_coarse, step)[:self.len_full], dims="date")  # Interpolate to daily log Rt
            r_t = pm.Deterministic("r_t", pt.exp(log_r_t), dims="date")  # Exponentiate to get Rt
            seed = pm.HalfNormal("seed", sigma=1e4)  # Adjusted prior for seed to handle absolute case numbers
            y0 = pt.zeros(self.len_full, dtype=ptconfig.floatX)  # Initial array for infections
            y0 = pt.set_subtensor(y0[0], seed)  # Set first day to seed

            #  infections(t) = Rt * инфекционный вклад прошлых дней
            def scan_fn(t, gt, y, r_t):  # Scan function for serial interval convolution
                return pt.set_subtensor(y[t], r_t[t] * pt.sum(y * gt))
            outputs, _ = scan(  # Use scan to compute infections over time
                fn=scan_fn,
                sequences=[pt.arange(1, self.len_full), convolution_ready_gt],
                outputs_info=y0,
                non_sequences=r_t,
                n_steps=self.len_full - 1,
            )
            infections = pm.Deterministic("infections", outputs[-1], dims="date")  # Infections time series
            #  свёртка с задержками → ожидаемые подтверждения
            infections_4d = pt.reshape(pt.cast(infections, ptconfig.floatX), (1, 1, 1, self.len_full))  # Reshape for conv
            p_delay_4d = pt.reshape(pt.as_tensor_variable(p_delay_np), (1, 1, 1, p_delay_np.shape[0]))  # Reshape delay
            test_adjusted_positive = pm.Deterministic(  # Convolve infections with delay to get expected cases (since no tests, this is expected positives)
                "test_adjusted_positive",
                conv2d(
                    infections_4d,
                    p_delay_4d,
                    border_mode="full",
                )[0, 0, 0, : self.len_full],
                dims="date"
            )

            #  Предполагаем постоянную ascertainment → exposure = 1
            exposure = pm.Deterministic("exposure", pt.ones(self.len_full), dims="date")  # Constant 1
            positive = pm.Deterministic("positive", exposure * test_adjusted_positive, dims="date")  # Expected positive cases
            ### Likelihood: Negative Binomial – учитываем сверхдисперсию
            alpha = pm.Gamma("alpha", mu=4, sigma=2)  # Dispersion parameter for Negative Binomial
            mu_hist = positive[:self.len_historical][nonzero_days] + 1e-10  # Mu for historical days
            pm.NegativeBinomial(  # Likelihood for observed positives
                "nonzero_positive",
                mu=mu_hist,
                alpha=alpha,
                observed=self.observed['positive'][nonzero_days].values,
                dims="nonzero_date",
            )
            p_ppc = pt.clip(alpha / (alpha + mu_hist), 1e-9, 1 - 1e-9)  # p for PPC Negative Binomial
            pm.NegativeBinomial(  # For posterior predictive checks
                "nonzero_positive_ppc",
                n=alpha,
                p=p_ppc,
                dims="nonzero_date",
            )

        return self.model  # Return built model

    # =========================================================
    # СЭМПЛИРОВАНИЕ ПОСТЕРИОРА
    # - По возможности используем ускоренный JAX-NUTS
    # =========================================================
    def sample(self, draws=10, tune=10, chains=4, target_accept=0.95, engine="jax"):
        """ Sample from the model.
        """
        if self.model is None:  # Build if not already
            self.build()
        start = {"log_r_coarse": np.full(self.n_steps, 0.0), "seed": 100.0, "alpha": 4.0}  # Good initial values to avoid issues, adjusted for scale
        if engine == "jax" and USE_JAX:  # Use JAX sampler if available
            idata = sample_numpyro_nuts(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                chain_method="vectorized",
                model=self.model,
                progressbar=True,
                random_seed=42,  # For reproducibility
            )
            self.trace = idata  # Store inference data
            return self
        else:  # Fallback to PyMC sampler
            with self.model:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=1,
                    target_accept=target_accept,
                    init="advi+adapt_diag",  # Use ADVI for better initialization
                    start=start,  # Provide starting point
                    compute_convergence_checks=True,
                    progressbar=True,
                    random_seed=42,  # For reproducibility
                )
            return self

# Function to process one country
def process_country(country, owid_df, train_end='2020-12-01', future_days=13):
    """ Process data for one country, train model, predict, and compare.
    """
    country_data = owid_df[owid_df['location'] == country].set_index('date')  # Filter and set date index
    # JHU data has no tests, so model adjusted accordingly; no 'new_tests'
    # Filter time window starting when new_cases > 100
    start_date = country_data[country_data['new_cases'] > 100].index.min()  # Find first date with >100 cases
    if pd.isnull(start_date):  # If no such date
        print(f"No data >100 for {country}")
        return None
    country_data = country_data.loc[start_date:train_end]  # Slice data
    observed = pd.DataFrame({  # Create observed DF
        'positive': country_data['new_cases'].fillna(0),  # New cases as positive
    })
    gm = GenerativeModel(country, observed, future_days=future_days)  # Initialize model
    gm.sample()  # Sample from model
    with gm.model:  # Posterior predictive sampling
        inference_data = pm.sample_posterior_predictive(
            gm.trace,
            var_names=["nonzero_positive_ppc", "positive"],
            progressbar=False,
            extend_inferencedata=True
        )
    # Rt summary: mean and HDI
    #  80% доверительный интервал Rt
    rt_summary = az.summary(inference_data.posterior['r_t'], hdi_prob=0.8)  # 80% HDI
    # Predictions: mean positive cases

    predicted_full = inference_data.posterior['positive'].mean(dim=['chain', 'draw'])  # Average over samples
    # Real data for comparison up to 14.12.2020
    real_future = owid_df[
        (owid_df['location'] == country) & (owid_df['date'] > train_end) & (owid_df['date'] <= '2020-12-14')
    ][['date', 'new_cases']]  # Extract real future cases
    # Save results
    rt_summary.to_csv(f'{country}_rt_summary.csv')  # Save Rt summary
    pd.DataFrame({'date': gm.full_index, 'predicted_cases': predicted_full}).to_csv(f'{country}_predicted_cases.csv', index=False)  # Save predictions
    real_future.to_csv(f'{country}_real_future.csv', index=False)  # Save real future
    # Plots
    plt.figure(figsize=(12, 6))  # Figure for Rt
    plt.plot(gm.full_index, rt_summary['mean'], label='Estimated Rt')  # Plot mean Rt
    plt.fill_between(gm.full_index, rt_summary['hdi_10%'], rt_summary['hdi_90%'], alpha=0.3)  # Fill HDI
    plt.axhline(1, ls='--', color='red')  # Critical line at 1
    plt.title(f'Rt for {country}')  # Title
    plt.savefig(f'{country}_rt_plot.png')  # Save plot
    plt.show()  # Show plot
    plt.figure(figsize=(12, 6))  # Figure for cases
    plt.plot(gm.full_index[:gm.len_historical], observed['positive'], label='Real historical cases')  # Historical real
    plt.plot(gm.full_index, predicted_full, label='Predicted cases')  # Predicted
    if not real_future.empty:  # If future data available
        plt.plot(real_future['date'], real_future['new_cases'], label='Real future cases')  # Real future
    plt.title(f'Predicted vs Real cases for {country}')  # Title
    plt.legend()  # Legend
    plt.savefig(f'{country}_cases_plot.png')  # Save
    plt.show()  # Show
    return rt_summary, predicted_full, real_future  # Return results

# Main: Load data and process countries
if __name__ == "__main__":
    # Download data from the provided JHU link
    owid_url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv"  # Updated URL as per user
    owid_response = requests.get(owid_url)  # Get response
    owid_df = pd.read_csv(StringIO(owid_response.text), parse_dates=['date'])  # Read to DF with date parsing
    countries = ['Russia', 'Italy', 'Germany', 'France']  # Countries to process
    for country in countries:  # Loop over countries
        print(f"Processing {country}...")  # Print status
        process_country(country, owid_df)  # Process