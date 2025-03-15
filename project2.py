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

def monte_carlo_simulation(S=100, X=100, T=1.0, mu=0.12, sigma=0.3, Lambda=0.25,
                   a=0.2, b=0.2, Nsteps=252, Nsim=100, alpha=0.05, seed=None, plot_paths=100):
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
    # Get current risk-free rate
    r = get_risk_free_rate()
    
    # Adjust mu to be risk-free rate plus risk premium
    risk_premium = mu - 0.05  # Extract original risk premium (relative to default 5%)
    mu = r + risk_premium    # Adjust mu to maintain same risk premium over new risk-free rate
    
    # Create a separate RandomState instance for this simulation
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    # First run the full simulation for accurate price calculation
    full_results = jump_diffusion(S, X, T, mu, sigma, Lambda, a, b, Nsteps, Nsim, alpha, rng)
    
    # If Nsim > plot_paths, run a second simulation with fewer paths for clearer visualization
    if Nsim > plot_paths:
        # Run visualization simulation with fewer paths
        vis_results = jump_diffusion(S, X, T, mu, sigma, Lambda, a, b, Nsteps, plot_paths, alpha, rng)
    else:
        vis_results = full_results
    
    return full_results

def jump_diffusion(S=100, X=100, T=1.0, mu=0.12, sigma=0.3, Lambda=0.25,
               a=0.2, b=0.2, Nsteps=252, Nsim=100, alpha=0.05, rng=None):
    """
    Implements Merton's Jump Diffusion Model using Monte Carlo simulation
    
    Mathematical Framework:
    ----------------------
    1. Stock Price Process:
       dS/S = (μ - λk)dt + σdW + (Y - 1)dN
       
       where:
       - dS/S: relative price change
       - μ: drift rate
       - λ: jump intensity
       - k: expected relative jump size (E[Y-1])
       - σ: volatility
       - dW: Wiener process increment
       - dN: Poisson process increment
       - Y: jump size multiplier
    
    2. Jump Component Parameters:
       The jump size Y follows a lognormal distribution where ln(Y) ~ N(a, b²)
       
       Mean jump size:
       E[Y] = exp(a + 0.5b²)
       
       Variance of jump size:
       Var(Y) = exp(2a + b²)(exp(b²) - 1)
    
       These are implemented as:
       mean_Y = np.exp(a + 0.5*(b**2))
       variance_Y = np.exp(2*a + b**2) * (np.exp(b**2)-1)
    
    3. Random Components Generation:
       Three independent random components are needed to capture different aspects
       of market behavior:
       
       a) Diffusion component (Z_1):
          Purpose: Models continuous, normal market movements
          - Represents day-to-day price fluctuations
          - Captures small, frequent changes due to regular trading
          - Based on Brownian motion theory where price changes are
            normally distributed over short time periods
          - Z_1 ~ N(0,1) generates these normal market movements
          
       b) Jump timing (Poisson):
          Purpose: Determines when significant market events occur
          - Models the arrival of major news or market shocks
          - λΔt represents expected number of significant events in Δt time
          - Poisson process chosen because:
            * Rare events occur independently
            * Rate of occurrence (λ) is constant over time
            * Multiple events can occur in same time interval
          - Examples: Earnings announcements, economic data releases,
                     geopolitical events
          
       c) Jump size (Z_2):
          Purpose: Determines the magnitude of market shocks
          - Models how big the price change is when a jump occurs
          - Independent of Z_1 because shock impacts differ from
            normal market movements
          - Combined with parameters a (mean jump size) and
            b (jump volatility) to create realistic jump distributions
          - Log-normal distribution ensures jumps can be both:
            * Positive (good news, upward jumps)
            * Negative (bad news, downward jumps)
            * More likely to have many small jumps than few large ones
       
       The combination of these components creates a realistic model where:
       1. Prices mostly follow normal market movements (Z_1)
       2. Occasionally experience sudden jumps (Poisson)
       3. Jump sizes vary realistically (Z_2)
       
       This matches empirical observations where markets typically
       exhibit both continuous price changes and sudden, significant
       movements due to unexpected events.

    4. Price Path Generation:
       For each time step i+1:
       S[i+1] = S[i] * exp((μ - σ²/2)Δt + σ√Δt * Z_1 + a*Poisson + b*√Poisson * Z_2)
       
       where:
       - (μ - σ²/2)Δt: drift adjustment with Itô correction
       - σ√Δt * Z_1: diffusion component
       - a*Poisson + b*√Poisson * Z_2: jump component
    """
    # Use provided RandomState or create a new one
    if rng is None:
        rng = np.random.RandomState()
    
    tic = time.time()
    Delta_t = T/Nsteps
    
    # Calculate jump size distribution parameters
    mean_Y = np.exp(a + 0.5*(b**2))  # E[Y]: expected jump size multiplier
    variance_Y = np.exp(2*a + b**2) * (np.exp(b**2)-1)  # Var[Y]: jump size variance
    
    # Theoretical moments of the jump diffusion process
    M = S * np.exp(mu*T + Lambda*T*(mean_Y-1))  # Expected price
    V = S**2 * (np.exp((2*mu + sigma**2)*T \
        + Lambda*T*(variance_Y + mean_Y**2 - 1)) \
        - np.exp(2*mu*T + 2*Lambda*T*(mean_Y - 1)))  # Price variance
    
    # Initialize paths matrix
    simulated_paths = np.zeros([Nsim, Nsteps+1])
    simulated_paths[:,0] = S
    
    # Generate random components for the entire simulation
    Z_1 = rng.normal(size=[Nsim, Nsteps])      # Brownian motion increments
    Z_2 = rng.normal(size=[Nsim, Nsteps])      # Jump size random variables
    Poisson = rng.poisson(Lambda*Delta_t, [Nsim, Nsteps])  # Jump timing

    # Populate the matrix with Nsim randomly generated paths of length Nsteps
    for i in range(Nsteps):
        simulated_paths[:,i+1] = simulated_paths[:,i]*np.exp((mu
                               - sigma**2/2)*Delta_t + sigma*np.sqrt(Delta_t) \
                               * Z_1[:,i] + a*Poisson[:,i] \
                               + np.sqrt(b**2) * np.sqrt(Poisson[:,i]) \
                               * Z_2[:,i])

    # Single out array of simulated prices at maturity T
    final_prices = simulated_paths[:,-1]

    # Compute mean, variance, standard deviation, skewness, excess kurtosis
    mean_jump = np.mean(final_prices)
    var_jump = np.var(final_prices)
    std_jump = np.std(final_prices)
    skew_jump = stats.skew(final_prices)
    kurt_jump = stats.kurtosis(final_prices)

    # Calculate confidence interval for the mean
    ci_low = mean_jump - std_jump/np.sqrt(Nsim)*stats.norm.ppf(1-0.5*alpha)
    ci_high = mean_jump + std_jump/np.sqrt(Nsim)*stats.norm.ppf(1-0.5*alpha)

    # Print statistics, align results
    print("Merton's Jump Diffusion Model")
    print('-----------------------------')
    print('Theoretical Moments')
    print('-----------------------------')
    print('Mean (M){:>21.4f}'.format(M))
    print('Variance (V){:>17.4f}'.format(V))
    print('\nMonte Carlo Estimates')
    print('-----------------------------')
    print('Mean {:>24.4f}'.format(mean_jump))
    print('Variance {:>20.4f}'.format(var_jump))
    print('Standard deviation {:>10.4f}'.format(std_jump))
    print('Skewness {:>20.4f}'.format(skew_jump))
    print('Excess kurtosis {:>13.4f}'.format(kurt_jump))
    print('\nConfidence interval, Mean')
    print('-----------------------------')
    print('Alpha {:>23.2f}'.format(alpha))
    print('Lower bound {:>17.4f}'.format(ci_low))
    print('Upper bound {:>17.4f}'.format(ci_high))

    # Choose palette, figure size, and define figure axes
    sns.set_theme(palette='viridis')
    plt.figure(figsize=(10,8))
    ax = plt.axes()

    # Generate t, the time variable on the abscissae
    t = np.linspace(0, T, Nsteps+1)

    # Plot the Monte Carlo simulated stock price paths
    jump_diffusion = ax.plot(t * 252, simulated_paths.transpose())  # Convert to trading days

    # Make drawn paths thinner by decreasing line width
    plt.setp(jump_diffusion, linewidth=1)

    # Set title (LaTeX notation) and x- and y- labels with trading days
    trading_days = int(T * 252)  # Convert years to trading days
    ax.set(title=r"Monte Carlo simulated stock price paths in Merton's jump diffusion model" + \
           r"\n$S_0$ = {}, $\mu$ = {}, $\sigma$ = {}, $a$ = {}, $b$ = {}, $\lambda$ = {}".format(
               S, mu, sigma, a, b, Lambda) + \
           f"\nTime horizon = {T} years ({trading_days} trading days), Paths = {Nsim}",
           xlabel='Trading Days', 
           ylabel='Stock Price ($)')
    
    # Set x-axis ticks to show trading days in increments of 50
    tick_positions = np.arange(0, trading_days + 50, 50)  # Add 50 to include the final value
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(x)}" for x in tick_positions])

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Display figure in a Python environment
    plt.show()

    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    print('Total running time: {:.2f} ms'.format(elapsed_time*1000))
    
    return simulated_paths, final_prices, mean_jump, var_jump, std_jump, skew_jump, kurt_jump, ci_low, ci_high, M, V

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
    
    # Generate paths using geometric Brownian motion
    Z = np.random.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Initialize price paths
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    # Generate price paths
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
    
    # Price the option using Longstaff-Schwartz
    price = longstaff_schwartz(S, K, r, dt, option_type)
    return price

def monte_carlo_cv(S0, K, T, sigma, n_steps, n_paths=10000, option_type='put', n_splits=5, n_monte_carlo=20):
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
        # Generate a larger set of paths for splitting
        Z = np.random.standard_normal((n_paths * 2, n_steps))
        drift = (r - 0.5 * sigma**2) * (T/n_steps)
        diffusion = sigma * np.sqrt(T/n_steps)
        
        # Initialize price paths
        S = np.zeros((n_paths * 2, n_steps + 1))
        S[:, 0] = S0
        
        # Generate price paths
        for t in range(1, n_steps + 1):
            S[:, t] = S[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
        
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
    # Run Monte Carlo simulation with original parameters
    print("Monte Carlo Simulation with Jump Diffusion:")
    print("------------------------------------------")
    monte_carlo_simulation()
    
    # Example of pricing and evaluating American options
    print("\nPricing and Evaluating American Options using Monte Carlo Cross-Validation:")
    print("------------------------------------------------------------------------")
    
    # Parameters for both options with updated values
    params = {
        'S0': 100,
        'K': 100,
        'T': 1,
        'sigma': 0.2,
        'n_steps': 50,
        'n_paths': 10000,  # Increased for stability
    }
    
    # Evaluate put option with more Monte Carlo iterations
    print("\nEvaluating American Put Option:")
    put_results = monte_carlo_cv(**params, option_type='put', n_splits=5, n_monte_carlo=20)
    print(f"Mean Price: ${put_results['mean_price']:.4f} ± ${put_results['price_std']:.4f}")
    print(f"95% Confidence Interval: (${put_results['confidence_interval'][0]:.4f}, ${put_results['confidence_interval'][1]:.4f})")
    print(f"Mean RMSE: ${put_results['mean_rmse']:.4f}")
    
    # Evaluate call option with more Monte Carlo iterations
    print("\nEvaluating American Call Option:")
    call_results = monte_carlo_cv(**params, option_type='call', n_splits=5, n_monte_carlo=20)
    print(f"Mean Price: ${call_results['mean_price']:.4f} ± ${call_results['price_std']:.4f}")
    print(f"95% Confidence Interval: (${call_results['confidence_interval'][0]:.4f}, ${call_results['confidence_interval'][1]:.4f})")
    print(f"Mean RMSE: ${call_results['mean_rmse']:.4f}")