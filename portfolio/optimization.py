"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import numpy as np
import scipy.optimize as sco

from util import get_data, plot_data
from analysis import get_portfolio_value, get_portfolio_stats


def find_optimal_allocations(prices):
    """Find optimal allocations for a stock portfolio, optimizing for Sharpe ratio.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio

    Returns
    -------
        allocs: optimal allocations, as fractions that sum to 1.0
    """
    # Number of allocations
    noa = np.shape(prices)[1]

    # Set the constraints: allocation weights sum to 1.0
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Set the bounds for the coefficients (all between 0 and 1.0)
    bnds = tuple((0, 1) for x in range(noa))

    # Provide an initial guess, based on equal weights, to the optimizer:
    initial_guess = noa * [1. / noa, ]

    opts = sco.minimize(min_func_sharpe, initial_guess, args=(prices,), method='SLSQP', bounds=bnds, constraints=cons)

    return opts['x']


def min_func_sharpe(weights, prices):
    port_val = get_portfolio_value(prices, weights)
    return -get_portfolio_stats(port_val)[3]


def optimize_portfolio(start_date, end_date, symbols):
    """Simulate and optimize portfolio allocations."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get optimal allocations
    allocs = find_optimal_allocations(prices)
    allocs = allocs / np.sum(allocs)  # normalize allocations, if they don't sum to 1.0

    # Get daily portfolio value (already normalized since we use default start_val=1.0)
    port_val = get_portfolio_value(prices, allocs)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # # Print statistics
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Symbols:", symbols)
    print("Optimal allocations:", allocs)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Volatility (stdev of daily returns):", std_daily_ret)
    print("Average Daily Return:", avg_daily_ret)
    print("Cumulative Return:", cum_ret)

    # Compare daily portfolio value with normalized SPY
    normed_SPY = prices_SPY / prices_SPY.ix[0, :]
    df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
    plot_data(df_temp, title="Daily Portfolio Value and SPY")


def test_run():

    start_date = '2010-01-01'
    end_date = '2010-12-31'
    symbols = ['GOOG', 'AAPL', 'GLD', 'HNZ']

    # Optimize portfolio
    optimize_portfolio(start_date, end_date, symbols)


def test_cases(allocs, refallocs):
    test_result = np.zeros(allocs.size, dtype=bool)
    if (np.array_equal(allocs, refallocs)):
        print("Tests Passed.")
        return

    # sum(allocations) = 1.0 +- 0.02
    # case1
    case1 = allocs.sum()
    if 0.98 <= case1 <= 1.02:
        test_result[0] = True

    # (abs(allocations)) = 1.0 +- 0.02
    # case2
    case2 = sum(abs(x) for x in allocs)
    if 0.98 <= case2 <= 1.02:
        test_result[1] = True

    # allocation is between 0 and 1.0 +- 0.02 (negative allocations are allowed if they are very small)
    # case3
    if (allocs >= -0.02).all() and (allocs <= 1.02).all():
        test_result[2] = True

    # case4
    # Each allocation matches reference solution +- 0.10

    if np.allclose(refallocs, allocs, atol=0.10):
        test_result[3] = True

    if False in test_result:
        raise ValueError(
            "Oops!  One or more values (as described in rubric) are not within the acceptable range.  Try again...")
    else:
        print("Tests Passed")
    return


if __name__ == "__main__":
    test_run()
