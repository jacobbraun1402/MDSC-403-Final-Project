import numpy as np
import pandas as pd
import os
np.random.seed(0)
from time import time
import tests_and_models.stats_tests as stats_tests

# function written by Google Gemini AI
# Response came up when I was trying to look up how to clear console output
def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

# Run the Mann-Whitney test for each SNP
def run_mann_whitney_tests(data):
    # Column 3 and beyond in the dataframe contains all the SNPs
    all_locs = data.columns.to_list()[2:]
    length=len(all_locs)
    results=[None for i in range(length)]
    for i, coord in enumerate(all_locs):
        if i % 100 == 0:
            clear_screen()
            print(f'Running Mann-Whitney Tests\nTest {i}/{length}')
        subs = data[['5_FT10', coord]]
        results[i]=(stats_tests.whitney_test(subs))
    # Keep track of all tests results in this dataframe
    # And remove rows where we had pvals that were NaN.
    # This happens when there are less than 2 groups
    return pd.DataFrame(results).dropna(subset=['pval'])

# Perform permutation test for every SNP in the dataset
def run_permutation_tests(data, test_stat):
    all_locs = data.columns.to_list()[2:]
    length=len(all_locs)
    perm_results = [None for i in range(length)]
    edfs = dict()
    for i, coords in enumerate(list(all_locs)):
        if i % 10 == 0: 
            clear_screen()
            print(f'Running Permutation Tests\nTest {i}/{length}')
        subs = data[['5_FT10', coords]]
        perm_results[i] = stats_tests.get_perm_test_results(subs, edfs, test_stat)
    return pd.DataFrame(perm_results).dropna(subset=['pval'])

if __name__=='__main__':
    start = time()
    print('Reading data')
    data = pd.read_csv('../data/processed.csv', index_col=0, header=None).transpose().astype({'5_FT10':np.float64})

    whitney_res = run_mann_whitney_tests(data)
    whitney_res = stats_tests.adjust_p_values(whitney_res)

    permutation_test_res = run_permutation_tests(data, stats_tests.mean_diff)
    permutation_test_res = stats_tests.adjust_p_values(permutation_test_res)
    permutation_test_res = permutation_test_res.rename(columns={'pval':'perm_pval', 'corrected_pval':'perm_corrected_pval'})
    
    merged_res=pd.merge(whitney_res, permutation_test_res, on='location')
    merged_res.to_csv('../data/stat_test_results.csv', index=False)
    duration = time() - start
    print(f'Total execution time: {(duration/60):.2f} minutes')