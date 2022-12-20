"""
Script for all plotting functions and anything tangentially related
"""
import pandas as pd
import matplotlib.patches as mpatches
from analysis.data_manips.data_extraction import ale

def cont_2_color(continent: str) -> str:
    transform = {"Africa": "r", "Americas": "b", "Asia": "g", "Europe": "m", "Oceania": "c"}
    return transform[continent]

def plot_life_exps(ax, df: pd.DataFrame, country: str, color: str=None, ec: str="k", alpha: float=0.5) -> None:
    """
    Plot all life expectencies
    """
    X, Y = ale(df, country, "year"), ale(df, country)
    ax.scatter(X, Y, c=color, ec=ec, alpha=alpha)
    return None

def labels_titles(ax, xlabel: str="", ylabel: str="", title: str="") -> None:
    """
    Add axis labels and titles
    """
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    return

def funky_legend(ax) -> None:
    """
    Creating legend for use with funky continental color scheme
    Go into 
    """
    conts = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
    patches = [mpatches.Patch(color=cont_2_color(cont), ec="k", label=f"{cont}") for cont in conts]
    ax.legend(handles=patches)
    return 

if __name__ == "__main__":
    pass