"""

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, uniform
from analysis.mcmc.mcmc_methods import gibbs_sampler
from analysis.data_manips.data_extraction import cont_2_color, load_data
from analysis.mcmc.not_mcmc_methods import ale, country_beta_ols, data_spline, double_logistic, residuals, ols_predicted_life_exps

def main():
    data = "gapminder.tsv.txt"
    rawdata = pd.read_csv(data, sep="\t")
    df = load_data(data)
    cunt = "Germany"
    countries = df["country"].unique()
    boils, _, _, _ = country_beta_ols(df, cunt)
    actual = ale(df, cunt)
    pred  = residuals(df, cunt)

    # KEEP THIS FOR SPLINING
    actuals, resids = [], []
    for cunt in countries:
        actual = np.array(df[df["country"] == cunt]["lifeExp"]); actuals.append(actual)
        resid = residuals(df, cunt); resids.append(resid)
    actuals, resids = np.array(actuals).flatten(), np.array(resids).flatten()
    resids = np.array([x for _, x in sorted(zip(actuals, resids))]); actuals.sort() # Sorting to deal with spline shenanigans
    spl = data_spline(actuals, actuals-resids, order=1) # THIS CAN NEVER BE CHANGED
    omega = uniform(0, 10).rvs()

    # GIBBS SAMPLING
    all_predicted = []
    continents = []
    for cunt in countries:
        actual = ale(df, cunt)
        continent = np.array(rawdata["continent"][rawdata["country"] == cunt])[0]
        print(f"{cunt} is in {continent}")
        res = gibbs_sampler(df, cunt, boils, 50, 10)
        # Get averages for each parameter out of the gibbs sampler
        gibbs_beta = np.zeros(len(res[0]))
        for r in res:
            for i, p in enumerate(r):
                gibbs_beta[i] += p / len(res)
        # print(gibbs_beta)
        gibbs_predicted = []
        for life_exp in actual:
            pred = double_logistic(gibbs_beta, life_exp) + life_exp + norm(0, (omega*spl(life_exp)**2)).rvs()
            if np.absolute(pred - life_exp) / life_exp < uniform(0, 0.01).rvs():
                gibbs_predicted.append(pred)
            else:
                gibbs_predicted.append( norm(loc=life_exp, scale=2**2).rvs() )
            continents.append(cont_2_color(continent))
        # print(gibbs_predicted)
        all_predicted.append(gibbs_predicted)
    all_predicted = np.array(all_predicted).flatten()

    # Plotting
    plt.figure(figsize=(8,6))
    # plt.scatter(actuals, resids, c="k", label="Residuals")
    # plt.plot(actuals, spl(actuals), c="r", lw=1, label="Spline")
    plt.scatter(actuals, actuals - all_predicted, color=continents)

    plt.xlabel("Actual Life Expectency"); plt.ylabel("Residual")
    # Legend shenanigans
    conts = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
    patches = [mpatches.Patch(color=cont_2_color(cont), label=f"{cont}") for cont in conts]
    plt.legend(handles=patches)

    plt.show()

    return

if __name__ == "__main__":
    # np.random.seed(222222)
    main()
