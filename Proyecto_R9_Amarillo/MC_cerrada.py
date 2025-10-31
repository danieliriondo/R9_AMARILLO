import numpy as np
from numpy.linalg import cholesky

np.random.seed(77)

def MonteCarlo(initial_maturity, S_0, num_sims, num_assets, num_asian_dates, value_date_index, correl_matrix,risk_free_rate, vols):
    '''
  Inputs:
  -------
  * S_0 (float): initial spot prices of the assets
  * num_asian_dates (int): number of asian dates. Notice that this includes the initial date, which fixes the initial spot values.
  * value_date_index (int): index, within the array of asian dates indicating the value date. (Actual step)
  * risk_free_rate (float): risk free rate
  * num_assets (int): number of underlying assets.
  * vols (array(float)): array of indiv assets vols.
  * correl_matrix (array(float, float)): matrix of correlations
  * initial_maturity (float): maturity of the product as seen on initial spot fixing date. (Years)
  * num_sims (int): number of simulations
  Outputs:
  --------
  * rets (numpy.ndarray): simulation of the evolution of the assets. Shape (num_sims, num_asian_dates-1, num_assets)
  * payoff (numpy.ndarray): Payoff of every simulation done. Shape (num_sims, )
  * premium (numpy.float64): present value in the actual step given by value_date_index.
  '''

    # Inputs:

    # Simulation of assets up to value date:

    init_time_array = np.linspace(0, initial_maturity, num_asian_dates)

    delta_t = initial_maturity / (num_asian_dates - 1)
    num_steps = value_date_index
    num_remaining_steps = num_asian_dates - value_date_index -1

    # Independent brownians
    inc_W = np.random.normal(size=(num_assets, num_steps), scale = np.sqrt(delta_t))

    # Cholesky matrix
    m = cholesky(correl_matrix)

    # Correlated brownians
    inc_W_correl = np.matmul(m, inc_W)

    # Independent brownian motion (the axis order is done to be able to correlate them with a matrix multiplication)
    inc_W_remaining = np.random.normal(size=(num_sims, num_remaining_steps, num_assets), scale = np.sqrt(delta_t))

    # We correlate them
    inc_W_correl_remaining = np.matmul(inc_W_remaining, m.T)

    # We transpose the 3D matrix of correlated B. motion (path, asset, time step)

    inc_W_correl_remaining = inc_W_correl_remaining.transpose([0,2,1])

    aux = np.repeat(inc_W_correl[None,...],num_sims,axis=0)

    # We attach the brownians obtained from t= 0 to value date

    inc_W_correl_total = np.concatenate((aux, inc_W_correl_remaining), axis = 2)

    # We compute exponential returns

    gross_rets_total = np.exp((risk_free_rate - 0.5 *vols **2) * delta_t + vols * inc_W_correl_total)

    # We simulate the underlyings

    S_T = np.cumprod(np.concatenate((np.repeat(S_0.reshape(-1,1)[None,...],num_sims,axis=0), gross_rets_total), axis = 2), axis = 2)

    # We compute the returns

    rets = S_T[:,:,1:] / np.repeat(S_0.reshape(-1,1)[None,...],num_sims,axis=0)

    payoff = np.maximum(np.sum(np.prod(rets, axis = (2))**(1/(num_assets * (num_asian_dates-1))), axis = 1)-3,0)

    # Calculate the premium as the discounted average value

    premium = np.mean(payoff*np.exp(-risk_free_rate*(initial_maturity-init_time_array[value_date_index])))

    rets = rets.transpose([0,2,1])

    return rets, payoff, premium

from scipy.stats import norm

N = norm.cdf

def Black(Forward, Strike, TTM, rate, Vol, IsCall):

  '''
  Inputs:
  -------
    Forward (float): Forward value
    Strike (float): strike price
    TTM (float): time to maturity in years
    rate (float): risk free rate
    Vol (float): volatility
    IsCall (bool): True if call option, False if put option
  Outputs:
  --------
    Option premium (float)
  '''

  if TTM >0:

    d1 = (np.log(Forward/Strike) + (Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))
    d2 = (np.log(Forward/Strike) + (- Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))

    if IsCall:

      return (Forward*N(d1)-Strike*N(d2))*np.exp(-rate*TTM)

    else:

      return (-Forward*N(-d1)+Strike*N(-d2))*np.exp(-rate*TTM)

  else:

    if IsCall:

      return np.maximum(Forward-Strike,0)

    else:

      return np.maximum(-Forward+Strike,0)
    

def BasketGeomAsian(num_asian_dates, value_date_index, risk_free_rate, num_assets, assets_vol, assets_correl, initial_maturity, price_history, IsCall):
  '''
  Inputs:
  -------
  * num_asian_dates (int): number of asian dates. Notice that this includes the initial date, which fixes the initial spot values.
  * value_date_index (int): index, within the array of asian dates indicating the value date.
  * risk_free_rate (float): risk free rate
  * num_assets (int): number of underlying assets.
  * assets_vol (array(float)): array of indiv assets vols.
  * assets_correl (array(float, float)): matrix of correlations
  * initial_maturity (float): maturity of the product as seen on initial spot fixing date.
  * price_history (array(float, float)): history of fixings of the underlyings up to value date. Assets per row, time steps per column.
  * IsCall (bool): True if call option, False if put option
  Outputs:
  --------
  * Option price (float)
  '''

  init_time_array = np.linspace(0, initial_maturity, num_asian_dates)

  pending_times_array = init_time_array[value_date_index+1:] - init_time_array[value_date_index]

  mu = np.sum(risk_free_rate - 0.5*assets_vol*assets_vol)*np.sum(pending_times_array) / (num_assets * (num_asian_dates-1))

  diag_vol = np.diag(assets_vol.reshape(-1))

  cov_matrix = np.matmul(diag_vol, np.matmul(assets_correl, diag_vol))

  xx, yy = np.meshgrid(pending_times_array, pending_times_array, sparse=True)
  z = np.minimum(xx, yy)

  V = np.sum(cov_matrix) * np.sum(z) / (num_assets*num_assets*(num_asian_dates-1)*(num_asian_dates-1))

  Forward = np.power(np.prod(price_history[:, 1:value_date_index+1] / price_history[:,0].reshape(-1,1)),1.0/(num_assets * (num_asian_dates-1)))

  Forward *= np.power(np.prod(price_history[:,value_date_index] / price_history[:,0]), (num_asian_dates-value_date_index-1)/(num_assets * (num_asian_dates-1)))

  Forward *= np.exp(mu + 0.5 * V)

  remaining_maturity = initial_maturity - init_time_array[value_date_index]


  return Black(Forward, 1.0, remaining_maturity, risk_free_rate,np.sqrt(V / remaining_maturity), IsCall)
