import numpy as np
from numpy import mean, std, sqrt
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad

# Number of permutations to run for each permutation test. Better p-values when there's 10000 iterations but very long computation time.
NUM_PERMUTATIONS = 10000

def normality_test(ft):
    """
    Run two tests for normality on the distribution of flowering times: the shapiro test and the Kolmogorov-Smirnov test.
    Both test the null hypothesis that the data is normally distributed.
    """
    ft_m = mean(ft)
    ft_sd = std(ft, ddof=1)
    cdf = lambda x: stats.norm.cdf(x, loc=ft_m, scale=ft_sd)

    ks_res = stats.ks_1samp(ft, cdf)
    print(f'KS test p-value: {ks_res.pvalue}')
    shapiro_res = stats.shapiro(ft)
    print(f'Shapiro test p-value: {shapiro_res.pvalue}')


def whitney_test(subset):
    '''
    Perform the Mann-Whitney U test for a SNP. Non-parametric counterpart to the t-test for independent samples.
    parameter subset: dataframe that contains column for the SNP of interest, and the flowering times of the samples.
    '''
    location = subset.columns.to_list()[-1]
    result = {'location': location,
           'A_mean':np.nan,
           'T_mean':np.nan,
           'C_mean':np.nan,
           'G_mean':np.nan,
           'pval': np.nan}
    
    # All SNPs in the dataset either have 1 or 2 different bases
    # So this is double checking that the SNP in question has 2 bases
    k = len(subset[location].unique())
    if k==2:
        g = subset[location].unique()
        # Get the flowering times for each group and perform the test
        x = subset.loc[subset[location] == g[0],'5_FT10'].values
        y = subset.loc[subset[location] == g[1],'5_FT10'].values
        result['pval'] = stats.mannwhitneyu(x, y).pvalue

        # Get the mean flowering time for each group
        result[f'{g[0]}_mean'] = mean(x)
        result[f'{g[1]}_mean'] = mean(y)
    return result

class empirical_distribution:
    '''
        Object representing the empirical distribution of a set of observations.
        Contains methods to find the probability density, as well as the cumulative density and survival function.
    '''
    def __init__(self, obs, bw=1):
        '''
        Constructs distribution from observations, and smooths it out using Gaussian kernel density estimation.
        parameter bw: smoothing bandwidth for KDE. A higher value results in a smoother function, but is less accurate, especially for multimodal distributions.
        '''
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(obs.reshape(-1,1))
        self.bounds = std(obs) * 10


    def pdf(self, x):
        '''
        Computes probability density at one or more points
        '''
        return np.exp(self.kde.score_samples(np.atleast_1d(x).reshape(-1,1)))
    
    def cdf(self, q):
        '''
        Compute cumulative probability P(q <= X) using Scipy's `quad` integration method on the PDF.
        '''
        return quad(self.pdf, q-self.bounds, q)[0]
    
    def sf(self, q):
        '''
        Calculate the value of survival function P(q >= X). Equivalent to 1-cdf(q), but this function is more precise.
        '''
        return quad(self.pdf, q, q+self.bounds)[0]
    
    # def two_tailed_p(self, q):
    #     return 2 * min(self.cdf(q), self.sf(q))

def mean_diff(x,y):
    return abs(mean(x) - mean(y))


def permutation_test(x, y, edfs: dict, stat, return_obs = False):
    """
    Run a permutation test that compares the absolute difference in mean flowering times for two groups. The two groups are the two different bases found at the SNP. The cdfs variable is a dictionary keeping track of CDFs that we can reuse, speeding up computation time greatly. The key of CDFs is the size of the first group, so if we're running another test where the size of the first group is already a key in the dictionary, we can just use the CDF associated with that key. Because each CDF is made from the same data, the CDF would look exactly the same as the already stored CDF if we were to find it again.
    """ 
    test_stat = stat(x,y)
    n_x = len(x)
    if n_x not in edfs:
        # Combine all observations into one array
        pooled = np.concat([x,y])
        sample_stats = np.zeros(shape=NUM_PERMUTATIONS)
        # sample_stats[-1] = test_stat
        N = len(pooled)
        for i in range(NUM_PERMUTATIONS):
            x_id = np.random.choice(N, n_x, replace=False)
            # assign n samples to group x and the rest to group y and calculate the difference in means
            x_sample = [pooled[j] for j in x_id]
            y_sample = [pooled[j] for j in range(N) if j not in x_id]
            sample_stats[i]=stat(x_sample, y_sample)
        # Create the continuous ECDF from the difference in means calculated from the permutations
        edf = empirical_distribution(sample_stats)
        edfs[n_x] = edf
    else:
        edf = edfs[n_x]
    if return_obs:
        return {'test_stat': test_stat, 'pvalue': edf.sf(test_stat), 'observations': sample_stats, 'edf': edf}
    else:
        return edf.sf(test_stat) 

def get_perm_test_results(df:pd.DataFrame, edfs, test_stat):
    """
    Compile permutation test results for a SNP.
    df format should be similar to that of subset parameter for Mann-Whitney test.
    """
    location = df.columns.to_list()[-1]
    perm_result = {'location': location, 'pval': np.nan}
    k = len(df[location].unique())
    s = [0,0]
    if k==2:
        for i,g in enumerate(df[location].unique()):
            group = df.loc[df[location] == g,:]
            s[i] = group['5_FT10'].values
        perm_result['pval'] = permutation_test(x=s[0], y=s[1], edfs=edfs, stat=test_stat)
    return perm_result


def adjust_p_values(results):
    """
    Since we're running over 200,000 tests on the same data, we need to adjust the p-values obtained from the test.
    I am adjusting using the Holm-Šidák method, which seems to be a good tradeoff between being conservative enough but still powerful.
    Formula obtained from: https://en.wikipedia.org/wiki/Holm–Bonferroni_method
    """
    results = results.sort_values(by='pval')
    pvals = results['pval'].to_numpy()
    m = len(pvals)
    pvals_corrected = np.array([0. for i in range(m)])
    pvals_corrected[0] = 1.-((1.-pvals[0])**(m-1))
    for i in range(1,len(pvals_corrected)):
        pvals_corrected[i] = max(pvals_corrected[i-1], 1.-((1.-pvals[i])**(m-i)))
    results['corrected_pval'] = pvals_corrected
    # Some p-values were so small that some of the decimal places got lost during calculations and the adjusted
    # p value became 0. When this happens, I just replace it with the original p-value so that
    # We can still make our manhattan plots.
    results.loc[results['corrected_pval'] == 0., 'corrected_pval'] = results.loc[results['corrected_pval'] == 0., 'pval']
    return results