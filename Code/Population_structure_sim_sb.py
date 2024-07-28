# %%

# This part of the code initializes various variables and data structures,
# including setting the assumptions for mosquito survival and defining the
# effects of interventions. It also sets up age classes for the mosquitoes.

import numpy as np
import pandas as pd
import datetime
import multiprocessing
from multiprocessing import Pool
import itertools

from time import time
from scipy.stats import chi2_contingency, wilcoxon
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

# Set the number of CPU cores to use for parallel processing
num_cores = multiprocessing.cpu_count()

# Set the date for file naming
date_today = datetime.date.today()

# Set EV (environmental variation) to True to run the simulations for Fig 4b,c,d and Table S4.
# Set EV to False to run the simulations for Fig S2.
EV = True

# Assumptions:
# Simulate a population with a survival rate of 0.91 (gambiae) or 0.82 (arabiensis).
# The 's' variable represents the daily survival rate.
# The 'p' variable represents the daily death probability before intervention.
# These values are chosen based on the mosquito species being gambiae.
s = {"gambiae": 0.91, "arabiensis": 0.82}["gambiae"]
p = 1 - s

# Define the effects of interventions.
# LLIN (Long-Lasting Insecticide Nets) results in a 4-fold increase in the death rate starting on day 4 of life.
# ATSB (Attractive Toxic Sugar Baits) results in a 3-fold increase in the death rate and works from day 1.
intervention_tab = pd.DataFrame({"effect": [1, 4, 2], "day.active": [2, 4, 2]})
intervention_tab.index = ["Control", "LLIN", "ATSB"]

# Define the maximum lifespan of mosquitoes and create a list of days.
n_day = 16
day = np.arange(1, n_day + 1)

# Define age classes: 1-4, 5-10, 11+
# age_cut = [min(day) - 0.5, 4.5, 10.5, max(day) + 0.5]
# age_bin = [day[(day >= age_cut[i - 1]) & (day < age_cut[i])] for i in range(1, len(age_cut))]

age_cut = [min(day) - 0.5, 9.5, max(day) + 0.5]
age_bin = [
    day[(day >= age_cut[i - 1]) & (day <= age_cut[i])] for i in range(1, len(age_cut))
]


# Create a dictionary to hold age bins.
age_bin_dict = {f"0-9": age_bin[0], f"10-16": age_bin[1]}

# Create a dictionary to map age group to a numeric value
age_group_mapping = {
    age_group: index + 1 for index, age_group in enumerate(age_bin_dict)
}

# Define age bin labels based on age bins
age_bin_labels = list(age_bin_dict.keys())

# Create a list of numeric values corresponding to age groups
age_bin_num = list(range(1, len(age_bin_labels) + 1))
print(age_bin_num)

# %%

# This part of the code calculates age structures based on intervention effects,
# creates plots to visualize the age structures, and reads confusion matrices from CSV files.
# The age structures are displayed using line plots and a stacked bar chart.

# Calculate age structure for the three groups based on intervention effects.
# The age structure is calculated as a probability distribution over days.

age_prob_list = {}
for intervention in intervention_tab.index:
    death_prob = np.array(
        [0]
        + [p] * (intervention_tab["day.active"][intervention] - 2)
        + [p * intervention_tab["effect"][intervention]]
        * (n_day - intervention_tab["day.active"][intervention] + 1)
    )
    age_prob = np.cumprod(1 - death_prob) / np.sum(np.cumprod(1 - death_prob))
    age_prob_dict = dict(zip(day, age_prob))
    age_prob_list[intervention] = age_prob_dict

age_prob_summary = pd.DataFrame(age_prob_list).describe()
print(age_prob_summary)


# %%
# # Convert the per-day age structure to binned age structure.
# age_prob_bin_list = {}
# for intervention in intervention_tab.index:
#     age_prob_bin = {age_class: np.sum([age_prob_list[intervention][str(age)] for age in age_bin])
#                     for age_class, age_bin in age_bin_dict.items()}
#     age_prob_bin_list[intervention] = age_prob_bin

age_prob_bin_list = {}
for intervention in intervention_tab.index:
    age_prob_bin = {
        age_class: sum(
            age_prob_list[intervention][age] for age in age_bin_dict[age_class]
        )
        for age_class in age_bin_dict.keys()
    }
    age_prob_bin_list[intervention] = age_prob_bin

age_prob_bin_summary = pd.DataFrame(age_prob_bin_list).describe()
print(age_prob_bin_summary)


# %%
# Plot age structures for the three intervention groups.

# Assuming age_prob_list is a dictionary
# Convert it into a DataFrame
age_prob_list_df = pd.DataFrame(age_prob_list)

# Then, you can concatenate the DataFrames
max_value = np.ceil(10 * max(age_prob_list_df.values.flatten())) / 10

# The rest of your plotting code remains the same
colors = ["black", "blue", "red"]

# Create 'day' variable from 1 to 20
day = list(range(1, 17))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_ylim(0, max_value)

# Plot age structures for the three intervention groups
ax1.plot(
    day,
    age_prob_list["Control"].values(),
    marker="o",
    linestyle="-",
    color=colors[0],
    label="Control",
)
ax1.plot(
    day,
    age_prob_list["LLIN"].values(),
    marker="o",
    linestyle="-",
    color=colors[1],
    label="LLIN",
)
ax1.plot(
    day,
    age_prob_list["ATSB"].values(),
    marker="o",
    linestyle="-",
    color=colors[2],
    label="ATSB",
)

# plt.gca().yaxis.tick_right()
# plt.gca().yaxis.set_label_position("right")
plt.ylabel("Proportion in population")
plt.xlabel("Mosquito age (days)")
plt.legend(loc="upper right", frameon=False)
plt.grid(False)
plt.title("Age Structure Comparison")

# # Add age class proportions to the plot.
# bin_scale = max(pd.concat(age_prob_bin_list).values) / max(pd.concat(age_prob_list).values)
# for age_class in age_bin_dict:
#     for intervention, age_prob in age_prob_bin_list.items():
#         plt.plot(list(age_bin_dict[age_class]), age_prob[age_class] / bin_scale, color=colors[intervention_tab.index.get_loc(intervention)],
#                  linestyle='-', linewidth=4, alpha=0.4)

# plt.gca().yaxis.tick_right()
# plt.gca().yaxis.set_label_position("right")
# plt.ylabel("Proportion in population")
# plt.xlabel("Mosquito age (days)")
# plt.legend(loc="upper right", frameon=False)
# plt.grid(True)
# plt.title("Age Structure Comparison")

# %%
# Create a bar chart of the age structure.
nice_colors = ["#e0f3db", "#43a2ca"]
age_prob_bin_df = pd.DataFrame(age_prob_bin_list).T
# age_prob_bin_df.index = [f'{age_class}d' for age_class in age_bin_dict]
ax = age_prob_bin_df.plot(
    kind="bar", stacked=False, legend=True, edgecolor="k", figsize=(6, 4), width=0.6
)
plt.ylabel("Proportion")
plt.xlabel("Age Classes")
plt.xticks(rotation=0)
plt.legend(loc="upper right", frameon=True)
plt.title("Binned Age Structure")

# Read confusion matrices from CSV files.
if EV:
    mat_tab = pd.read_csv(
        "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\Fold\Training_Folder_selected_wns\cm_2.csv",
        header=None,
    )
else:
    mat_tab = pd.read_csv(
        "Confusion_Matrices/confusion_matrices_0_05_2020-01-22.csv", header=None
    )

mat_tab.columns = [f"r{row}c{col}" for row in range(1, 3) for col in range(1, 3)]

if EV:
    mat_tab["n.tcv"] = 209
else:
    mat_tab["n.tcv"] = range(1, len(mat_tab) + 1)


# %% Define a function to simulate data and perform tests

# def simulate_data(j, age_bin_num, mat_tab, mat_names):
#     np.random.seed(assumptions.at[j, "rand.seed"])

#     simres_list = []

#     for i in range(assumptions.at[j, "nsim"]):
#         n = assumptions.loc[j, "n"]

#         # Simulate data with true age in days
#         dat = pd.concat(
#             [
#                 pd.DataFrame(
#                     {
#                         "intervention": intervention,
#                         "age": np.random.choice(
#                             np.arange(1, 17),
#                             n,
#                             p=list(age_prob_list[intervention].values()),
#                         ),
#                     }
#                 )
#                 for intervention in intervention_tab.index
#             ]
#         )

#         dat["age.cat"] = pd.cut(
#             dat["age"], bins=age_cut, labels=[1, 2]
#         )  # Replace '0-9' and '10-16' labels with 1 and 2
#         # print('dat',   pd.cut(dat['age'], bins=age_cut, labels=[0, 1])

#         # Handle the case where mat_tab has only one row
#         if mat_tab.shape[0] == 1:
#             mat = mat_tab.iloc[0, :-1].values.reshape(2, 2)
#         else:
#             mat = mat_tab.loc[
#                 assumptions.at[j, "mat.row"], mat_tab.columns[:-1]
#             ].values.reshape(2, 2)

#         dat["age.cat.est"] = dat["age.cat"].apply(
#             lambda a: np.random.choice(age_bin_num, p=mat[int(a) - 1])
#         )

#         # out_list = []
#         # for h, intervention in enumerate(intervention_tab.index, start=1):
#         #     dat_test = dat[dat['intervention'].isin([intervention])]
#         #     print(Counter(dat_test['intervention']))

#         for intervention in intervention_tab.index:
#             dat_test = dat[dat["intervention"].isin(intervention_tab.index)]
#             # print(Counter(dat_test['intervention']))

#             contingency_table = pd.crosstab(
#                 dat_test["age.cat"], dat_test["intervention"]
#             )
#             # print(contingency_table)

#             # # Perform Wilcoxon-Mann-Whitney test
#             wilcox = wilcoxon(dat_test["age.cat"], correction = True)
#             wilcox_est = wilcoxon(dat_test["age.cat.est"], correction = True)

#             wil_p_value, wil_p_value_est = wilcox.pvalue, wilcox_est.pvalue

#             # Handle cases where p-values are NaN (similar to is.na() in R)
#             if np.isnan(wil_p_value):
#                 wil_p_value = 0
#             if np.isnan(wil_p_value_est):
#                 wil_p_value_est = 0

#             # Perform Chi-squared test
#             xtab = pd.crosstab(dat_test["age.cat.est"], dat_test["intervention"])
#             # print(xtab)
#             # chi_p_value = scipy.stats.chi2_contingency(contingency_table).pvalue < 0.05
#             # chi_p_value_est = scipy.stats.chi2_contingency(xtab).pvalue < 0.05

#             # Perform Chi-squared test
#             chi_val = chi2_contingency(contingency_table)
#             chi_value_est = chi2_contingency(xtab)

#             # Export test results
#             epsilon = 1e-7  # small constant to protect from div by 0
#             out = {
#                 f"wil.pow.{intervention}": wil_p_value,
#                 f"wil.pow.est.{intervention}": wil_p_value_est,
#                 f"chi.pow.{intervention}": chi_val.pvalue,
#                 f"chi.pow.est.{intervention}": chi_value_est.pvalue,
#                 f"prop.control.{intervention}": xtab[intervention_tab.index[0]].values
#                 / (xtab.sum(axis=1) + epsilon),
#                 f"prop.intervention.{intervention}": xtab[intervention].values
#                 / (xtab.sum(axis=1) + epsilon),
#             }

#             # print(out)

#             simres_list.append(pd.Series(out))

#     simres_tab = pd.concat(simres_list, axis = 0)

#     print(f"{round(100 * (j+1) / assumptions.shape[0])}% complete")

#     return simres_tab


# # Define the age bin numeric values
# age_bin_num = np.arange(1, len(age_bin) + 1)

# # Define the scenarios and assumptions
# mat_row_values = np.arange(1, mat_tab.shape[0] + 1)
# n_values = [20, 50, 100, 150, 200, 250, 300]
# repeated_mat_row = np.repeat(np.arange(1, mat_tab.shape[0] + 1), len(n_values))
# repeated_nsim = [10000] * (len(n_values) * len(mat_row_values))

# assumptions = pd.DataFrame(
#     {
#         "mat.row": repeated_mat_row,
#         "n": n_values * len(mat_row_values),
#         "nsim": repeated_nsim,
#     }
# )

# # Set random seeds
# global_rand_seed = 78212
# # np.random.seed(global_rand_seed)

# assumptions["global.rand.seed"] = global_rand_seed
# assumptions["rand.seed"] = np.random.choice(np.arange(1, int(1e5), len(assumptions)))


#%%

# %% Define a function to simulate data and perform tests

def simulate_data(j, age_bin_num, mat_tab, mat_names):
    np.random.seed(assumptions.at[j, "rand.seed"])

    simres_list = []

    for i in range(assumptions.at[j, "nsim"]):
        n = assumptions.loc[j, "n"]

        # Simulate data with true age in days
        dat = pd.concat(
            [
                pd.DataFrame(
                    {
                        "intervention": intervention,
                        "age": np.random.choice(
                            np.arange(1, 17),
                            n,
                            p=list(age_prob_list[intervention].values()),
                        ),
                    }
                )
                for intervention in intervention_tab.index
            ]
        )

        dat["age.cat"] = pd.cut(
            dat["age"], bins=age_cut, labels=[1, 2]
        )  # Replace '0-9' and '10-16' labels with 1 and 2
        # print('dat',   pd.cut(dat['age'], bins=age_cut, labels=[0, 1])

        # Handle the case where mat_tab has only one row
        if mat_tab.shape[0] == 1:
            mat = mat_tab.iloc[0, :-1].values.reshape(2, 2)
        else:
            mat = mat_tab.loc[
                assumptions.at[j, "mat.row"], mat_tab.columns[:-1]
            ].values.reshape(2, 2)

        dat["age.cat.est"] = dat["age.cat"].apply(
            lambda a: np.random.choice(age_bin_num, p=mat[int(a) - 1])
        )

        out_list = []
        # for h, intervention in enumerate(intervention_tab.index, start=1):
        #     dat_test = dat[dat['intervention'].isin([intervention])]
        #     print(Counter(dat_test['intervention']))

        # Iterate over rows of intervention.tab starting from the second row (h=2 to nrow(intervention.tab))
        for h in range(len(intervention_tab)):
            # Filter dat for relevant interventions
            dat_test = dat[dat['intervention'].isin([intervention_tab.index[0], intervention_tab.index[h]])]
            # print(dat_test)
            print('data test', dat_test['intervention'].unique())
            # print('unique age cat estimates', dat_test["age.cat.est"].nunique())

            # Perform a Wilcoxon-Mann-Whitney test to compare age distributions
            # Assuming age.cat and age.cat.est are columns in dat
            wilcat = wilcoxon(dat_test['age.cat'], zero_method='zsplit', correction = True)
            wilcat_est = wilcoxon(dat_test['age.cat.est'], zero_method='zsplit', correction = True)
            wil_pow = wilcat.pvalue
            wil_pow_est = wilcat_est.pvalue

            # Check if wil_pow or wil_pow_est are NaN and replace them with 0
            if np.isnan(wil_pow):
                wil_pow = 0
            if np.isnan(wil_pow_est):
                wil_pow_est = 0

            # Perform a Chi-squared test to compare age distributions
            # Assuming age.cat and age.cat.est are columns in dat
            xtab = pd.crosstab(pd.Categorical(dat_test['age.cat.est'], categories=[1, 2, 3]), dat_test['intervention'])
            chi_cat = chi2_contingency(pd.crosstab(dat_test['age.cat'], dat_test['intervention']))
            chi_cat_est = chi2_contingency(xtab[xtab.sum(axis=1) > 0])
            chi_pow = chi_cat.pvalue 
            chi_pow_est = chi_cat_est.pvalue

            # Calculate proportions
            prop_control = xtab / xtab.sum(axis=0).loc[intervention_tab.index[0]]
            prop_intervention = xtab / xtab.sum(axis=0).loc[intervention_tab.index[h]]

            # Store test results in a dictionary
            results = {
                'wil.pow': wil_pow,
                'wil.pow.est': wil_pow_est,
                'chi.pow': chi_pow,
                'chi.pow.est': chi_pow_est,
                'prop.control': prop_control[intervention_tab.index[0]],
                'prop.intervention': prop_intervention[intervention_tab.index[h]]
            }

            # Rename dictionary keys with intervention names
            results = {f'{key}.{intervention_tab.index[h]}': value for key, value in results.items()}

            # Append the results dictionary to the out_list
            out_list.append(results)
            # print(out_list)

        # Convert the list of dictionaries into a flattened dictionary
        out_dict = {key: value for result in out_list for key, value in result.items()}
        # print('out dict', out_dict)
        # print(out_dict.keys())

        out_list_df = pd.DataFrame(out_dict)
        simres_list.append(out_list_df)
        # print('out_list_df', out_list_df)

    simres_tab = pd.concat(simres_list, axis = 1)

    print(f"{round(100 * (j+1) / assumptions.shape[0])}% complete")

    return simres_tab


# Define the age bin numeric values
age_bin_num = np.arange(1, len(age_bin) + 1)

# Define the scenarios and assumptions
mat_row_values = np.arange(1, mat_tab.shape[0] + 1)
n_values = [20, 50, 100, 150, 200, 250, 300]
repeated_mat_row = np.repeat(np.arange(1, mat_tab.shape[0] + 1), len(n_values))
repeated_nsim = [100] * (len(n_values) * len(mat_row_values))

assumptions = pd.DataFrame(
    {
        "mat.row": repeated_mat_row,
        "n": n_values * len(mat_row_values),
        "nsim": repeated_nsim,
    }
)

# Set random seeds
global_rand_seed = 78212
# np.random.seed(global_rand_seed)

assumptions["global.rand.seed"] = global_rand_seed
assumptions["rand.seed"] = np.random.choice(np.arange(1, int(1e5), len(assumptions)))


# %% Single cores, avoiding multicore processing

start_time = time()

results = [
    simulate_data(j, age_bin_num, mat_tab, mat_tab.columns) for j in assumptions.index
]

# Combine results from all scenarios
simres_tab = pd.concat(results, axis = 1)

# # %% Version for multiple cores
# # Define a function to be run in parallel
# start_time = time()
# def simulate_data_wrapper(j):
#     return simulate_data(j, age_bin_num, mat_tab, mat_tab.columns)

# # Create a pool of workers
# with multiprocessing.Pool() as p:
#     results = p.map(simulate_data_wrapper, assumptions.index)

# # Combine results from all scenarios
# simres_tab = pd.concat(results, axis=1)

# Calculate the mean across simulations for each scenario
# simres_tab = simres_tab.apply(pd.to_numeric, errors = "coerce")
# power_estimates = simres_tab.mean(axis = 1)

# Print results
# print(power_estimates)

end_time = time()
print("Run time : {} s".format(end_time - start_time))
print("Run time : {} m".format((end_time - start_time) / 60))
print("Run time : {} h".format((end_time - start_time) / 3600))

# %%

# Create a loop to iterate over the rows of intervention_tab
for i in range(1, len(intervention_tab)):
    gp = intervention_tab.index[i]
    form = f'wil.pow.est.{gp} ~ n'
    form2 = f'wil.pow.{gp} ~ n'
    ntcv_lev = np.unique(out['n.tcv'])
    ntcv_col = ntcv_col[0:len(ntcv_lev)] if not EV else ntcv_col
    ntcv_col_dict = {lev: col for lev, col in zip(ntcv_lev, ntcv_col)}

    powercurve_file = f'agestructure.powercurve.{"" if EV else "var."}{s}.{gp}.{datetime.date.today()}.pdf'

    plt.figure(figsize=(8/2.54, 7/2.54), dpi=100)
    old_par = plt.rcParams.copy()

    plt.ylim(0, 1)
    plt.xlim(min(out['n']), max(out['n']) * 1.20**(0 if not EV else 1))
    plt.xlabel("N per population")
    plt.ylabel("Power")

    plt.xticks(unique(out['n']), rotation=45)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)

    for ntcv in ntcv_lev:
        df = out[out['n.tcv'] == ntcv]
        plt.plot(df['n'], df[form], marker='o', markersize=4, label=f'ntcv = {ntcv}', color=ntcv_col_dict[ntcv])

    max_power = out.groupby('n')[f'wil.pow.{gp}'].mean()
    if EV:
        plt.plot(max_power.index, max_power, linestyle='--', label='Max Power', color='black')
        plt.legend(loc='bottomright')
    else:
        # Calculate some statistics when EV is False
        subset = out[out['n'] == 20]
        stdev_values = subset.filter(like=f'wil.pow.est', axis=1).std().values
        power_values = subset.filter(like=f'wil.pow.est', axis=1).mean().values
        unique_nsim = len(np.unique(out['nsim']))
        stats = stdev_values / np.sqrt(power_values * (1 - power_values) / unique_nsim)

    plt.title(gp)
    plt.rcParams.update(old_par)
    plt.savefig(powercurve_file)
    plt.close()