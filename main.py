"""

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import norm, uniform
from analysis.mcmc.mcmc_methods import gibbs_sampler
from analysis.visuals.plotting import cont_2_color, funky_legend, plot_life_exps, labels_titles
from analysis.data_manips.data_extraction import ale, load_data, get_all_continents, get_continent, grab_life_exp_gain
from analysis.mcmc.not_mcmc_methods import country_beta_ols, data_spline, double_logistic, linear_fit_params_life_exps, ols_predicted_life_exps, residuals

def main():
    data = "gapminder.tsv.txt"
    df = load_data(data)
    count = "Croatia"
    rawdata = pd.read_csv(data, sep="\t")
    countries = df["country"].unique()
    beta_ols, _, _, _ = country_beta_ols(df, count)
    actual = ale(df, count)
    pred  = residuals(df, count)

    # Plotting raw data
    plt.figure(figsize=(8,6))
    X = np.array([])
    mS = {"Africa": [], "Americas": [], "Asia": [], "Europe": [], "Oceania": []}
    bS = {"Africa": [], "Americas": [], "Asia": [], "Europe": [], "Oceania": []}
    rS = {"Africa": [], "Americas": [], "Asia": [], "Europe": [], "Oceania": []}
    for count in countries:
        continent = get_continent(rawdata, count)
        plot_life_exps(plt.gca(), df, count, color=cont_2_color(continent), ec=None, alpha=0.5)
        # Linear regression
        m, b, r = linear_fit_params_life_exps(df, count)
        mS[continent].append(m); bS[continent].append(b); rS[continent].append(r)
        X = np.append(X, ale(df, count, "year"))
    # Lines of best fit for each continent
    m_per_cont = [np.mean(series) for series in mS.values()]
    b_per_cont = [np.mean(series) for series in bS.values()]
    r_per_cont = [np.mean(series) for series in rS.values()]
    for cont, m, b, r in zip(["Africa", "Americas", "Asia", "Europe", "Oceania"], m_per_cont, b_per_cont, r_per_cont):
        f = lambda x: m*x + b
        print(f"Average correlation coefficient {cont} is {r:0.4f}")
        plt.plot(X, f(X), c=cont_2_color(cont), lw=1.5, path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()], label=f"Best fit for {cont}")
    labels_titles(plt.gca(), xlabel="Year", ylabel="Average Life Expectancy")
    funky_legend(plt.gca())
    # plt.legend()
    plt.show()

    # KEEP THIS FOR SPLINING
    actuals, resids, continents = [], [], np.array([])
    for count in countries:
        actual = ale(df, count); actuals.append(actual)
        resid = residuals(df, count); resids.append(resid)
        continent = get_continent(rawdata, count); continents = np.append(continents, [cont_2_color(continent) for _ in range(len(actual))])
    actuals, resids = np.array(actuals).flatten(), np.array(resids).flatten()
    resids = np.array([x for _, x in sorted(zip(actuals, resids))]); actuals.sort() # Sorting to deal with spline shenanigans
    spl = data_spline(actuals, actuals-resids, order=1) # THIS CAN NEVER BE CHANGED
    omega = uniform(0, 10).rvs()

    # # GIBBS SAMPLING FOR ALL COUNTRIES
    # all_predicted = []
    # continents = []
    # for count in countries:
    #     actual = ale(df, count)
    #     continent = np.array(rawdata["continent"][rawdata["country"] == count])[0]
    #     print(f"{count} is in {continent}")
    #     res = gibbs_sampler(df, count, beta_ols, 50, 10)
    #     # Get averages for each parameter out of the gibbs sampler
    #     gibbs_beta = np.zeros(len(res[0]))
    #     for r in res:
    #         for i, p in enumerate(r):
    #             gibbs_beta[i] += p / len(res)
    #     #
    #     gibbs_predicted = []
    #     for life_exp in actuals:
    #         pred = double_logistic(gibbs_beta, life_exp) + life_exp + norm(0, (omega*spl(life_exp)**2)).rvs()
    #         if np.absolute(pred - life_exp) / life_exp < 0.5:
    #             gibbs_predicted.append(pred)
    #         else:
    #             gibbs_predicted.append( norm(loc=life_exp, scale=2**2).rvs() )
    #         continents.append(cont_2_color(continent))
    #     # print(gibbs_predicted)
    #     all_predicted.append(gibbs_predicted)
    # all_predicted = np.array(all_predicted).flatten()

    # # Plotting raw life expectancy gains
    # actual_lexp = []
    # plt.figure(figsize=(8,6))
    # for count in countries:
    #     # continent = np.array(rawdata["continent"][rawdata["country"] == count])[0]
    #     le, leg = grab_life_exp_gain(df, count) # Actual life expectancy and actual gain to life expectancy
    #     continent = get_continent(rawdata, count)
    #     if max(np.absolute(leg)) < 5:
    #         plt.scatter(le, leg, c=cont_2_color(continent), ec="k")
    #     if count == "Croatia":
    #         actual_lexp.append(le)
    # # Superimposing gain function on top
    # # actual_lexp = np.array(actual_lexp).flatten()
    # # Y = double_logistic(gibbs_beta, actual_lexp)
    # # plt.plot(actual_lexp, Y, c="k", lw=2)
    # labels_titles(plt.gca(), xlabel="Life Expectancy", ylabel="Expected Gain")
    # funky_legend(plt.gca())
    # plt.show()

    # Plotting
    plt.figure(figsize=(8,6))
    plt.scatter(actuals, resids, c=continents, ec="k", label="Residuals")
    plt.plot(actuals, spl(actuals), c="k", lw=2, label="Spline")
    # plt.scatter(actuals, all_predicted, color=continents, ec="k")
    labels_titles(plt.gca(), xlabel="Actual Life Expectancy", ylabel="Predicted Life Expectancy")
    funky_legend(plt.gca())
    plt.show()

    return

if __name__ == "__main__":
    # np.random.seed(222222)
    main()
