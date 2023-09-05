import pandas as pd
import numpy as np
import matplotlib as mpl
import pygad
import numpy as np
import itertools
import random
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams["figure.figsize"] = (10, 7)
data = pd.read_csv("data_excerpt.csv")
data
df = data.copy()
start_date = "2021-04-01"
end_date = "2022-03-01"
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
df = df[["ticker", "date", "Close"]].rename(columns={"Close": "price"}).reset_index(drop=True)
tickers = df["ticker"].unique()
tickers_map = {i : j for i,j in zip(tickers, range(len(tickers)))}
tickers_map_reverse = {j : i for i,j in zip(tickers, range(len(tickers)))}

df["ticker_index"]  = df["ticker"].map(tickers_map)
firsts = (df.groupby('ticker').transform('first'))
df["adj_price"] = df["price"] / firsts["price"]
df = df[["ticker", "ticker_index", "date", "adj_price"]]

## Drop OGN as it joined SP500 midway
df = df[df["ticker"] != "OGN"]
df.groupby("ticker").count().sort_values("date")
# Genetic algorithm
# Below we try to find the best combination of 10 stocks that give the best return over the defined time period

## Define fitness function and utils

def portfolio_generate(df, tickers):
    portfolio = df[df['ticker_index'].isin(tickers)]
    portfolio = portfolio.groupby("date", as_index=False).sum()
    portfolio = portfolio.sort_values("date")
    return portfolio

def portfolio_return(portfolio):
    first_price = portfolio["adj_price"].iloc[0]
    last_price = portfolio["adj_price"].iloc[-1]
    return last_price / first_price - 1

def portfolio_risk(portfolio):
    portfolio["daily_change"] = portfolio["adj_price"].diff(1)
    portfolio["daily_change"] = portfolio["daily_change"] / portfolio["adj_price"]

    return portfolio["daily_change"].std()

def fitness_func(solution, solution_idx):
    portfolio = portfolio_generate(df, solution)
    ret = portfolio_return(portfolio)
    ris = portfolio_risk(portfolio)
    fitness = ret / ris
    return fitness

def visualize(df, solution):
    solution_fitness = fitness_func(solution, None)
    portfolio  = portfolio_generate(df, solution)
    portfolio["adj_price"] = (portfolio["adj_price"] / portfolio["adj_price"].iloc[0] ) * 100
    ax = portfolio.plot.line(x="date", y="adj_price")
    ax.set_ylim(90, 190)
    ret = round(portfolio_return(portfolio) * 100, 1)
    ris = round(portfolio_risk(portfolio) * 100, 1)
    
    print(f"Parameters of the best solution : {[tickers_map_reverse[i] for i in solution]}")
    print(f"Return: {ret}%")
    print(f"Risk: {ris}%")
    print(f"Risk adjusted return = {round(solution_fitness,1)}%")
## Define Genetic Algorithm

fitness_function = fitness_func

num_generations = 30
num_genes = 10

sol_per_pop = 90
num_parents_mating =  50

init_range_low = 0
init_range_high = 497
gene_type = int

parent_selection_type = "sss"
keep_parents = 30

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 30
## Initiate and run genetic algorithm

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_type=gene_type,
                       allow_duplicate_genes=False,
                       random_seed=2)
ga_instance.run()
for i,j in zip (ga_instance.best_solutions, ga_instance.best_solutions_fitness):
  print([(tickers_map_reverse[k],k) for k in sorted(i)],j)
## Plot training, best resuls

ga_instance.plot_fitness(save_dir="result.png")
[solution, _, __] = ga_instance.best_solution()
visualize(df, solution)
## Top 10 performers benchmark

firsts = df.groupby("ticker_index", as_index=False).first()
firsts = firsts.rename({"adj_price": "first_price"}, axis=1)[["ticker_index", "first_price"]]
lasts = df.groupby("ticker_index", as_index=False).last()
lasts = lasts.rename({"adj_price": "last_price"}, axis=1)[["ticker_index", "last_price"]]

df_ = firsts.merge(lasts, on="ticker_index", how="left")
df_["return"] = df_["last_price"] / df_["first_price"]
df_ = df_.sort_values("return", ascending=False)
best_return = df_.head(10)["ticker_index"].unique()

visualize(df, best_return)
## S&P 500 benchmark

visualize(df, df["ticker_index"].unique())