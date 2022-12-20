"""

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.mcmc.mcmc_methods import gibbs_sampler
from analysis.data_manips.data_extraction import load_data
from analysis.mcmc.not_mcmc_methods import country_beta_ols, data_spline, residuals, ols_predicted_life_exps

def main():
    data = "gapminder.tsv.txt"
    df = load_data(data)
    cunt = "Afghanistan"
    countries = df["country"].unique()
    boils, _, _, _ = country_beta_ols(df, cunt)
    # pred = ols_predicted_life_exps(res, boils, cunt)
    actual = np.array(df[df["country"] == cunt]["lifeExp"])
    pred  = residuals(df, "Afghanistan")
    # res = gibbs_sampler("f", "f", "f", 1000, 100)


    actuals, resids = [], []
    for cunt in countries:
        actual = np.array(df[df["country"] == cunt]["lifeExp"]); actuals.append(actual)
        resid = residuals(df, cunt); resids.append(resid)
    actuals, resids = np.array(actuals).flatten(), np.array(resids).flatten()
    resids = np.array([x for _, x in sorted(zip(actuals, resids))]); actuals.sort() # Sorting to deal with spline shenanigans
    # print(resids.size)
    spl = data_spline(actuals, actuals-resids, order=1) # THIS CAN NEVER BE CHANGED
    # Plotting
    plt.figure(figsize=(8,6))
    plt.scatter(actuals, resids, c="k", label="Residuals")
    plt.plot(actuals, spl(actuals), c="r", lw=1, label="Spline")

    plt.xlabel("Actual Life Expectency"); plt.ylabel("Absolute Residuals")
    plt.legend()

    plt.show()

    return

if __name__ == "__main__":
    main()
