import pandas as pd

if __name__=='__main__':
    genotype = pd.read_csv('genotype.csv').astype({'Chromosome': str, 'Positions': str})
    phenotype = pd.read_table('FT10.txt', sep='\t').dropna().astype({'ecotype_id': str})

    # Find flower samples that are in both datasets, and filter the datasets based on the samples found
    sample_intersect = list(set(phenotype['ecotype_id']) & set(genotype.columns[2:]))
    phenotype_filtered = phenotype.loc[phenotype['ecotype_id'].isin(sample_intersect),:].reset_index(drop=True)
    genotype_filtered = genotype.loc[:,['Chromosome', 'Positions']+sample_intersect]

    # Transpose genotype dataframe so that the rows
    # Become the columns
    genotype_T = genotype_filtered.transpose()

    # extract ecotype ids from genotype dataframe
    ids = genotype_T.index[2:].to_list()

    # Create dictionary that will hold all of the data for our reformatted genotype
    # dataframe
    geno_dict = {'id':ids}
    # Iterate through all SNPs in the dataset
    # Currently, the first two rows are taken up by the Chromosome and Position values,
    # While all the rows below that will have the SNP values for each sample
    for i in genotype_T.columns:
        col = genotype_T[i]
        # Combine SNP chromosome and position into one string, which will then be used as the column name
        location = str(col.iloc[0]) + ':' + str(col.iloc[1])
        # Get all SNP values for the current SNP that we're on
        vals = col.iloc[2:].to_list()
        geno_dict[location] = vals

    genotype_final = pd.DataFrame(geno_dict)
    # Merge the reformatted genotype dataframe with the phenotype dataframe, using the ecotype id to match up rows
    merged = pd.merge(phenotype_filtered, genotype_final, left_on='ecotype_id', right_on='id').drop(columns=['id'])
    # Write the transpose of the dataframe to csv. For some reason it takes like 2 minutes to load the dataframe
    # When it has 200000 columns but like 2 seconds to load the transpose and transpose it again
    merged.transpose().to_csv('processed.csv', index=True, header=False)