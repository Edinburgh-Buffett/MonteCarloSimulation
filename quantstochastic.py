import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import random as rn
from scipy import stats
import scipy.stats as si
import seaborn as sns
import time 
from datetime import datetime
from fredapi import Fred
import os

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def get_risk_free_rate():
    """
    Get the current risk-free rate from FRED using 3-Month Treasury Bill rate
    Returns rate as a decimal (e.g., 0.05 for 5%)
    """
    try:
        # Initialize FRED with API key
        fred_key = '7ca637540188723e3026839a2cad1190'  # Your FRED API key
        fred = Fred(api_key=fred_key)
        
        # Get the 3-Month Treasury Bill rate (TB3MS)
        tb3ms = fred.get_series('TB3MS')
        
        # Get the most recent rate and convert to decimal
        current_rate = tb3ms.iloc[-1] / 100.0
        
        print(f"Current 3-Month Treasury Bill Rate: {current_rate:.4%}")
        return current_rate
        
    except Exception as e:
        print(f"Error fetching risk-free rate from FRED: {e}")
        print("Using default risk-free rate of 5%")
        return 0.05

def monte_carlo_simulation(S=100, X=100, T=1.0, mu=0.12, sigma0=0.3, Lambda=0.25,
                           a=0.2, b=0.2, Nsteps=252, Nsim=100, alpha=0.05, seed=None, plot_paths=100,
                           kappa=2.0, theta=0.09, xi=0.3, rho=-0.5):
    """
    Monte Carlo simulation for Merton's Jump Diffusion Model

    Default Parameter Choices:
    -------------------------
    1. Price Parameters (S=100, X=100):
       - S=100: Standard initial stock price for easy percentage calculations
       - X=100: At-the-money strike price to observe both upward and downward movements
       - These values allow clear observation of relative price changes
    
    2. Time and Return Parameters:
       - T=1.0: One year horizon, standard in financial analysis
       - mu=0.12 (12%): Composed of:
         * Risk-free rate (typically 2-5%)
         * Equity risk premium (typically 5-7%)
         * Additional premium for jump risk
       - sigma=0.3 (30%): Typical stock volatility in normal market conditions
         * Individual stocks: 20-40%
         * Market indices: 15-25%
         * Higher volatility accounts for jump component
    
    3. Jump Process Parameters:
       - Lambda=0.25: Average of one jump every 4 years
         * Rare enough to be "jumps" rather than normal fluctuations
         * Frequent enough to impact price dynamics
       - a=0.2: Mean jump size parameter
         * Positive value indicates upward bias in jumps
         * Reflects empirical observation that markets trend upward
       - b=0.2: Jump volatility parameter
         * Similar scale to diffusion volatility
         * Creates realistic jump magnitudes
    
    4. Simulation Parameters:
       - Nsteps=252: Number of trading days in a year
         * Provides daily granularity
         * Standard in financial markets
       - Nsim=100: Balance between:
         * Computational efficiency
         * Statistical significance
       - alpha=0.05: 95% confidence level
         * Standard in statistical analysis
       - plot_paths=100: Clear visualization without overcrowding
    
    These parameters are calibrated to:
    1. Match empirical market observations
    2. Provide realistic price dynamics
    3. Balance computational efficiency
    4. Enable clear visualization and analysis
    
    Mathematical Framework:
    ----------------------
    1. Stock Price Process:
       The stock price follows a jump-diffusion process with both continuous and discrete components:
       
       dS/S = (μ - λk)dt + σdW + (Y - 1)dN
       
       This can be decomposed into:
       a) Continuous component: (μ - λk)dt + σdW
          - Drift term: (μ - λk)dt accounts for expected return adjusted for jumps
          - Diffusion term: σdW represents normal market fluctuations
       
       b) Jump component: (Y - 1)dN
          - Y: random jump size multiplier
          - dN: Poisson jump indicator
    
    2. Jump Size Distribution:
       The jump sizes follow a lognormal distribution:
       ln(Y) ~ N(a, b²)
       
       Expected jump size: k = E[Y - 1] = exp(a + b²/2) - 1
       Jump variance: var(Y) = exp(2a + b²)(exp(b²) - 1)
    
    3. Expected Stock Price:
       E[S(T)] = S₀exp(μT + λT(E[Y]-1))
       
       This accounts for:
       - Continuous growth: exp(μT)
       - Jump effect: exp(λT(E[Y]-1))
    
    4. Price Variance:
       var[S(T)] = S₀²[exp((2μ + σ²)T + λT(var(Y) + E[Y]² - 1)) 
                      - exp(2μT + 2λT(E[Y]-1))]
       
       Components:
       - Diffusion variance: σ²T
       - Jump size uncertainty: var(Y)
       - Jump timing uncertainty: λT
    
    5. Discretization:
       The continuous-time process is discretized into Nsteps intervals:
       Δt = T/Nsteps
       
       Each step evolution:
       S(t+Δt) = S(t)exp((μ - σ²/2)Δt + σ√Δt·Z₁ + ln(Y)·dN)
       
       where:
       - Z₁ ~ N(0,1): standard normal for diffusion
       - dN ~ Poisson(λΔt): jump occurrence
       - ln(Y) = a + b·Z₂: jump size, Z₂ ~ N(0,1)
    """
    
    """
    Monte Carlo Simulation using Merton's Jump Diffusion Model with Stochastic Volatility
    """
    # Get current risk-free rate
    r = get_risk_free_rate()
    
    # Adjust mu to reflect the correct risk premium
    risk_premium = mu - 0.05  # Extract original risk premium (relative to default 5%)
    mu = r + risk_premium  # Adjust mu to maintain same risk premium

    # Use a separate RandomState instance
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # ** Run full simulation with stochastic volatility **
    full_results = jump_diffusion_stochastic_volatility(S, X, T, mu, sigma0, Lambda, a, b,
                                                        Nsteps, Nsim, alpha, kappa, theta, xi, rho, rng)
    
    # ** If Nsim > plot_paths, run a visualization simulation with fewer paths **
    if Nsim > plot_paths:
        vis_results = jump_diffusion_stochastic_volatility(S, X, T, mu, sigma0, Lambda, a, b,
                                                           Nsteps, plot_paths, alpha, kappa, theta, xi, rho, rng)
    else:
        vis_results = full_results

    return full_results
def jump_diffusion_stochastic_volatility(S=100, X=100, T=1.0, mu=0.12, sigma0=0.3, Lambda=0.25,
                                         a=0.2, b=0.2, Nsteps=252, Nsim=100, alpha=0.05,
                                         kappa=2.0, theta=0.09, xi=0.3, rho=-0.5, rng=None):
    """
    Merton's Jump Diffusion Model with Stochastic Volatility (Heston Model)
    """

    if rng is None:
        rng = np.random.RandomState()  # Ensure reproducibility

    start_time = time.time()  # Start timer

    Delta_t = T / Nsteps  # Time step

    # Initialize stock price & volatility paths
    S_paths = np.zeros((Nsim, Nsteps + 1))
    V_paths = np.zeros((Nsim, Nsteps + 1))
    S_paths[:, 0] = S
    V_paths[:, 0] = sigma0**2  # Initial variance

    # Generate correlated random numbers
    Z1 = rng.normal(size=(Nsim, Nsteps))  # Brownian motion for stock price
    Z2 = rng.normal(size=(Nsim, Nsteps))  # Brownian motion for volatility
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # Correlated with Z1

    for i in range(Nsteps):
        St = S_paths[:, i]
        Vt = V_paths[:, i]

        # Heston stochastic volatility update
        V_paths[:, i + 1] = np.maximum(Vt + kappa * (theta - Vt) * Delta_t + xi * np.sqrt(Vt * Delta_t) * Z2[:, i], 0)

        # Jump diffusion process
        jumps = rng.poisson(Lambda * Delta_t, Nsim)  # Poisson-distributed jumps
        Y = np.exp(a + b * rng.normal(size=Nsim))  # Jump sizes
        Jump_effect = (Y - 1) * jumps  # Jump impact

        # Stock price update
        S_paths[:, i + 1] = St * np.exp((mu - 0.5 * Vt) * Delta_t + np.sqrt(Vt * Delta_t) * Z1[:, i]) * (1 + Jump_effect)

    # Extract final stock prices
    final_prices = S_paths[:, -1]

    # Monte Carlo Statistics
    mean_price = np.mean(final_prices)
    variance_price = np.var(final_prices)
    std_dev_price = np.std(final_prices)
    skewness_price = stats.skew(final_prices)
    kurtosis_price = stats.kurtosis(final_prices)
    
    # Confidence interval
    ci_low = mean_price - std_dev_price / np.sqrt(Nsim) * stats.norm.ppf(1 - 0.5 * alpha)
    ci_high = mean_price + std_dev_price / np.sqrt(Nsim) * stats.norm.ppf(1 - 0.5 * alpha)

    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

    # Print statistics
    print("\nMerton’s Jump Diffusion with Stochastic Volatility")
    print("------------------------------------------------------")
    print(f"Mean:               {mean_price:.4f}")
    print(f"Variance:           {variance_price:.4f}")
    print(f"Standard Deviation: {std_dev_price:.4f}")
    print(f"Skewness:           {skewness_price:.4f}")
    print(f"Kurtosis:           {kurtosis_price:.4f}")
    print("\nConfidence Interval for Mean:")
    print(f"Alpha:              {alpha:.2f}")
    print(f"Lower Bound:        {ci_low:.4f}")
    print(f"Upper Bound:        {ci_high:.4f}")
    print("\n------------------------------------------------------")
    print(f"Total Running Time: {elapsed_time:.2f} ms")

    # Plot Monte Carlo simulated paths
    sns.set_theme(palette="viridis")
    plt.figure(figsize=(10, 6))
    plt.plot(S_paths[:100, :].T, alpha=0.5)  # Plot 100 sample paths
    plt.title("Monte Carlo Simulated Stock Price Paths (Jump Diffusion + Stochastic Volatility)")
    plt.xlabel("Time Steps (Trading Days)")
    plt.ylabel("Stock Price")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

    return {
        "paths": S_paths,
        "final_prices": final_prices,
        "mean": mean_price,
        "variance": variance_price,
        "std_dev": std_dev_price,
        "skewness": skewness_price,
        "kurtosis": kurtosis_price,
        "ci_low": ci_low,
        "ci_high": ci_high
    }

# In the main section, add an example of running the simulation
# if __name__ == "__main__":
#     # Example parameters for the Heston model
#     S0 = 100  # Initial stock price
#     K = 100  # Strike price
#     T = 1.0  # Time to maturity
#     mu = 0.12  # Drift
#     sigma_0 = 0.2  # Initial volatility
#     kappa = 2.0  # Mean reversion rate for volatility
#     theta = 0.2  # Long-term mean of volatility
#     eta = 0.2  # Volatility of volatility
#     rho = -0.5  # Correlation between asset price and volatility
#     Nsteps = 252  # Number of steps (trading days)
#     Nsim = 1000  # Number of simulated paths

#     print("Monte Carlo Simulation with Heston Stochastic Volatility Model:")
#     print("--------------------------------------------------------------")
#     jump_diffusion_stochastic_volatility(S=S0, X=K, T=T, mu=mu, sigma_0=sigma_0, kappa=kappa, theta=theta, eta=eta, rho=rho,
#                  Nsteps=Nsteps, Nsim=Nsim)
def longstaff_schwartz(paths, strike, r, dt, option_type='put'):
    """
    Implements the Longstaff-Schwartz algorithm for American option pricing
    
    Mathematical Framework:
    ----------------------
    1. Option Payoff:
       - Put: max(K - S, 0)
       - Call: max(S - K, 0)
       where K is strike price and S is stock price
    
    2. Continuation Value:
       - Estimated using regression: E[V(t+1)|S(t)]
       - Basis functions: [1, S, S²]
       - Regression equation: V(t+1) = β₀ + β₁S + β₂S²
    
    3. Exercise Decision:
       - Exercise if immediate payoff > continuation value
       - Value function: V(t) = max(immediate payoff, discounted continuation value)
    
    Algorithm Steps:
    ---------------
    1. Initialize terminal payoffs
    2. Backward recursion through time:
       a. Identify in-the-money paths
       b. Estimate continuation value using regression
       c. Compare with immediate exercise value
       d. Update option values
    3. Discount optimal exercise values
    
    Parameters:
    -----------
    paths : 
        Matrix of simulated stock price paths (n_paths × n_steps)
    strike : 
        Strike price of the option
    r : 
        Risk-free interest rate
    dt : 
        Time step size
    option_type : 
        'put' or 'call'
    
    Returns:
    --------
    American option price
    """
    n_paths, n_steps = paths.shape[0], paths.shape[1]-1
    df = np.exp(-r*dt)  # discount factor
    
    # Initialize cash flow matrix
    if option_type == 'put':
        h = np.maximum(strike - paths, 0)
    else:  # call option
        h = np.maximum(paths - strike, 0)
    
    # Initialize value matrix
    V = h.copy()
    
    # Backward recursion through time
    for t in range(n_steps-1, 0, -1):
        # Only consider paths that are in-the-money
        if option_type == 'put':
            itm = paths[:, t] < strike
        else:
            itm = paths[:, t] > strike
        
        if sum(itm) > 0:
            # Regression step
            X = paths[itm, t]
            Y = V[itm, t+1] * df
            
            # Use polynomial basis functions
            basis = np.column_stack([np.ones_like(X), X, X**2])
            beta = np.linalg.lstsq(basis, Y, rcond=None)[0]
            
            # Calculate continuation value
            C = np.dot(basis, beta)
            
            # Exercise decision
            exercise = h[itm, t]
            V[itm, t] = np.where(exercise > C, exercise, V[itm, t+1] * df)
            V[~itm, t] = V[~itm, t+1] * df
    
    # Optimal exercise at t=0
    return np.mean(np.maximum(h[:, 0], V[:, 1] * df))

def price_american_option(S0, K, T, sigma, n_steps, n_paths, option_type='put'):
    """
    Prices American options using Monte Carlo simulation and Longstaff-Schwartz method
    
    Mathematical Framework:
    ----------------------
    1. Stock Price Process (GBM):
       dS = rSdt + σSdW
       Discretized form:
       S(t+dt) = S(t)exp((r - σ²/2)dt + σ√dt * Z)
       where:
       - r: risk-free rate
       - σ: volatility
       - Z: standard normal random variable
    
    2. Path Generation:
       - Generate n_paths independent price paths
       - Each path has n_steps time steps
       - Use antithetic variates for variance reduction
    
    3. Option Valuation:
       - Apply Longstaff-Schwartz algorithm
       - Account for early exercise opportunity
       - Use regression for continuation value
    
    Parameters:
    -----------
    S0 : 
        Initial stock price
    K : 
        Strike price
    T : 
        Time to maturity
    sigma : 
        Volatility
    n_steps : 
        Number of time steps
    n_paths : 
        Number of simulation paths
    option_type : 
        'put' or 'call'
    
    Returns:
    --------
    American option price
    """
    # Get current risk-free rate
    r = get_risk_free_rate()
    
    # Input validation
    if S0 <= 0 or K <= 0:
        raise ValueError("Stock price and strike price must be positive")
    if T <= 0:
        raise ValueError("Time to maturity must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("Number of steps and paths must be positive")
    if option_type not in ['put', 'call']:
        raise ValueError("Option type must be 'put' or 'call'")
    
    dt = T/n_steps
    
    # Simulate paths using jump diffusion with stochastic volatility
    full_results = monte_carlo_simulation(S=S0, X=K, T=T, mu=r, sigma0=sigma, Lambda=0.25, a=0.2, b=0.2,
                                          Nsteps=n_steps, Nsim=n_paths, alpha=0.05, plot_paths=100)
    
    # Retrieve simulated paths
    S_paths = full_results["paths"]
    
    # Price the option using Longstaff-Schwartz
    price = longstaff_schwartz(S_paths, K, r, dt, option_type)
    return price

def monte_carlo_cv(S0, K, mu, Lambda,a,b, T, sigma, n_steps, n_paths=10000, option_type='put', n_splits=5, n_monte_carlo=20):
    """
    Performs Monte Carlo Cross-Validation for robust option pricing
    
    Methodology:
    -----------
    1. K-Fold Cross-Validation:
       - Split paths into k training and validation sets
       - Train on k-1 folds, validate on remaining fold
       - Repeat for all k folds
    
    2. Monte Carlo Iterations:
       - Repeat cross-validation multiple times
       - Generate new paths for each iteration
       - Calculate statistics across iterations
    
    3. Performance Metrics:
       - Mean price and standard deviation
       - RMSE between training and validation
       - 95% confidence intervals
    
    Statistical Measures:
    -------------------
    1. Price Statistics:
       - Mean: μ = (1/n)Σᵢpᵢ
       - Std Dev: σ = √[(1/n)Σᵢ(pᵢ-μ)²]
    
    2. Error Metrics:
       - RMSE = √[(1/n)Σᵢ(train_priceᵢ - val_priceᵢ)²]
    
    3. Confidence Intervals:
       - 95% CI: [2.5th percentile, 97.5th percentile]
    
    Parameters:
    -----------
    [Same as price_american_option with additional]
    n_splits : int
        Number of cross-validation folds
    n_monte_carlo : int
        Number of Monte Carlo iterations
    
    Returns:
    --------
    dict: Performance metrics and statistics
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    # Get current risk-free rate
    r = get_risk_free_rate()
    
    all_prices = []
    all_errors = []
    all_std_devs = []
    
    for mc_iter in range(n_monte_carlo):
         # Simulate paths using jump diffusion with stochastic volatility
        full_results = monte_carlo_simulation(S=S0, X=K, T=T, mu=r, sigma0=sigma, Lambda=Lambda, a=a, b=b,
                                              Nsteps=n_steps, Nsim=n_paths, alpha=0.05, plot_paths=100)
        
        # Retrieve simulated paths
        S = full_results["paths"]
        # K-fold Cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold_prices = []
        
        for train_idx, val_idx in kf.split(S):
            # Split paths into training and validation sets
            train_paths = S[train_idx]
            val_paths = S[val_idx]
            
            # Price option using training paths
            train_price = longstaff_schwartz(train_paths, K, r, T/n_steps, option_type)
            
            # Price option using validation paths
            val_price = longstaff_schwartz(val_paths, K, r, T/n_steps, option_type)
            
            fold_prices.append((train_price, val_price))
        
        # Calculate metrics for this Monte Carlo iteration
        train_prices, val_prices = zip(*fold_prices)
        mean_price = np.mean(val_prices)
        std_dev = np.std(val_prices)
        rmse = np.sqrt(np.mean((np.array(train_prices) - np.array(val_prices))**2))
        
        all_prices.append(mean_price)
        all_errors.append(rmse)
        all_std_devs.append(std_dev)
    
    # Aggregate results
    results = {
        'mean_price': np.mean(all_prices),
        'price_std': np.std(all_prices),
        'mean_rmse': np.mean(all_errors),
        'rmse_std': np.std(all_errors),
        'mean_std_dev': np.mean(all_std_devs),
        'confidence_interval': (
            np.percentile(all_prices, 2.5),
            np.percentile(all_prices, 97.5)
        )
    }
    
    return results

# Modify the main section to include MCCV evaluation
if __name__ == "__main__":
    print("Monte Carlo Simulation with Stochastic Volatility and Jump Diffusion:")
    print("------------------------------------------")
    
    # Parameters for both options with updated values
    params = {
        'S0': 100,
        'K': 100,
        'T': 1,
        'sigma': 0.2,  # Initial volatility (stochastic)
        'mu': 0.05,     # Risk-free rate
        'Lambda': 0.25, # Jump intensity (if using jumps)
        'a': 0.1,       # Mean reversion speed for volatility
        'b': 0.2,       # Long-term mean volatility level
        'n_steps': 50,
        'n_paths': 10000,  # Increased for stability
    }
    
    # Example of pricing and evaluating American options with stochastic volatility
    print("\nPricing and Evaluating American Put Option with Stochastic Volatility:")
    put_results = monte_carlo_cv(**params, option_type='put', n_splits=5, n_monte_carlo=20)
    print(f"Mean Price: ${put_results['mean_price']:.4f} ± ${put_results['price_std']:.4f}")
    print(f"95% Confidence Interval: (${put_results['confidence_interval'][0]:.4f}, ${put_results['confidence_interval'][1]:.4f})")
    print(f"Mean RMSE: ${put_results['mean_rmse']:.4f}")
    
    print("\nPricing and Evaluating American Call Option with Stochastic Volatility:")
    call_results = monte_carlo_cv(**params, option_type='call', n_splits=5, n_monte_carlo=20)
    print(f"Mean Price: ${call_results['mean_price']:.4f} ± ${call_results['price_std']:.4f}")
    print(f"95% Confidence Interval: (${call_results['confidence_interval'][0]:.4f}, ${call_results['confidence_interval'][1]:.4f})")
    print(f"Mean RMSE: ${call_results['mean_rmse']:.4f}")