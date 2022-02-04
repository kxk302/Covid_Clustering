import json
import numpy as np
import os
import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

pd.set_option('display.max_columns', None)

# When calculating the distance between 2 probability densities,
# if one probability value is 0 (or very small), the cross entropy
# (distance) value would be infinity. This brakes the DBSCAN algorithm.
# Replace infinity values with a large number, say 200.00
max_distance = 200.00

# KL divergence value of two identical distributions is 0
# Somehow it's calculated as NaN. Replace Nan with 0
min_distance = 0.00

# x axis values for calculating/plotting KDE of sample AF
x_idx = np.linspace(0.00, 1.00, num=100).tolist()


def get_kde_values(row):
    return gaussian_kde(row['AF']).evaluate(x_idx).tolist()


def get_kl_div(x, y):
    return entropy(x, y)


# Get the sample IDs and collection dates from 'input_file' for study
# accessions specified in 'study_accessions_file'. Filter the samples
# such that 'collection_dates' falls between 'start_date' and 'end_date'
def get_samples(input_file='gx-surveillance.json',
                study_accessions_file='Estonia_study_accessions.csv',
                start_date=None,
                end_date=None):
    input = None
    study_accessions = None
    samples = []
    collection_dates = []

    with open(input_file, 'r') as f:
        input = json.load(f)

    with open(study_accessions_file, 'r') as f:
        study_accessions = f.read().splitlines()

    total_sa_count = 0
    total_collection_dates_count = 0
    df = pd.DataFrame({'Sample': [],
                       'Collection_Date': []})

    for study_accession in study_accessions:
        sa_count = 0
        collection_dates_count = 0

        # Key would be Batch, value would be associated dictionary
        for (key, value) in input.items():
            if value['study_accession'] == study_accession:
                # Get the list of samples
                samples.extend(value['samples'])
                sa_count += len(value['samples'])

                # Get the list of collection dates for the samples
                collection_dates.extend(value['collection_dates'])
                collection_dates_count += len(value['collection_dates'])
        total_sa_count += sa_count
        total_collection_dates_count += collection_dates_count

        df_loop = pd.DataFrame({'Sample': samples,
                                'Collection_Date': collection_dates})                                

        print('Before filter. df_loop.shape: {}'.format(df_loop.shape))

        if start_date is not None:
            df_loop = df_loop[df_loop.Collection_Date >= start_date]
        if end_date is not None:
            df_loop = df_loop[df_loop.Collection_Date <= end_date]

        print('After filter. df_loop.shape: {}'.format(df_loop.shape))

        df = pd.concat([df, df_loop])
        print('df.shape: {}'.format(df.shape))

    return df


def preprocess(file_name, sep="\t", samples_df=None):

    # Read the input file. Select only the needed columns.
    df = pd.read_csv(file_name, sep)[['Sample', 'AF']]
    df_in = df.copy()

    # Clean up data by removing rows where af is greater than 1.0
    print('\n')
    print('Removing rows with AF greater than 1.0')
    df_in = df_in[df_in.AF <= 1.00]

    # sample stats
    print('Stats before filtering\n')
    print('Number of unique samples {}'.format(df_in['Sample'].nunique()))
    print('sample minimum: {}'.format(df_in['Sample'].min()))
    print('sample maximum: {}'.format(df_in['Sample'].max()))

    # af stats
    print('\n')
    print('Number of unique af {}'.format(df_in['AF'].nunique()))
    print('af minimum: {}'.format(df_in['AF'].min()))
    print('af maximum: {}'.format(df_in['AF'].max()))

    # Only keep samples specified in samples_df
    if samples_df is not None:
        df_in = df_in[df_in['Sample'].isin(samples_df['Sample'])]

    # sample stats
    print('Stats after filtering\n')
    print('Number of unique samples {}'.format(df_in['Sample'].nunique()))
    print('sample minimum: {}'.format(df_in['Sample'].min()))
    print('sample maximum: {}'.format(df_in['Sample'].max()))

    # af stats
    print('\n')
    print('Number of unique af {}'.format(df_in['AF'].nunique()))
    print('af minimum: {}'.format(df_in['AF'].min()))
    print('af maximum: {}'.format(df_in['AF'].max()))

    # Pivot the data frame and generate a list of AF for each sample
    df_piv = pd.pivot_table(df_in, index='Sample', values='AF', aggfunc=list)
    print('df_piv.head(5)')
    print(df_piv.head(5))
    print('df_piv.shape')
    print(df_piv.shape)

    # Clean up data by removing rows where af list has only one or two element
    # KDE calculation errors out for those
    # df_piv_clean = df_piv[ df_piv.AF.str.len() > 2]

    # Calculate
    df_piv['KDE_vals'] = df_piv.apply(get_kde_values, axis=1)

    print('df_piv.head(5)')
    print(df_piv.head(5))
    print('df_piv.shape')
    print(df_piv.shape)

    return df_piv


# eps:
#   The maximum distance between two samples for one to be considered as in
#   the neighborhood of the other. This is the most important DBSCAN parameter
#   to choose appropriately for your data set and distance function.
# min_samples:
#   The number of samples n a neighborhood for a point to be considered as
#   a core point. This includes the point itself.
# metric:
#   The metric to use when calculating distance between instances in a
#   feature array.
# metric_params:
#  Additional keyword arguments for the metric function.
def dbscan_clustering(file_name,
                      sep='\t',
                      eps=0.5,
                      min_samples=5,
                      metric='euclidean',
                      metric_params=None,
                      distances_file_name=None,
                      n_jobs=1,
                      input_file='gx-surveillance.json',
                      study_accessions_file='Estonia_study_accessions.csv',
                      start_date=None,
                      end_date=None):

    samples_df = get_samples(input_file=input_file,
                             study_accessions_file=study_accessions_file,
                             start_date=start_date,
                             end_date=end_date)

    df_piv_clean = preprocess(file_name=file_name, sep=sep, samples_df=samples_df)

    if metric == 'precomputed':
        distances = pd.read_csv(distances_file_name, sep=sep, index_col=0)

        # Replace infinity values in distances matrix with a large value
        # Replace NaN values (diagonal values) with 0
        # Replace negative values with 0
        distances.replace([np.inf], max_distance, inplace=True)
        distances.replace([np.nan], min_distance, inplace=True)
        distances[distances < 0] = 0.00

        # Run DBSCAN clustering algorithm on precomputed distance matric
        db = DBSCAN(eps=eps, min_samples=min_samples,
                    metric=metric, metric_params=metric_params,
                    n_jobs=n_jobs).fit(distances)
    else:
        # Run DBSCAN clustering algorithm
        db = DBSCAN(eps=eps, min_samples=min_samples,
                    metric=metric, metric_params=metric_params,
                    n_jobs=n_jobs).fit(df_piv_clean.KDE_vals.tolist())

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('\n')
    print('Number of clusters: {}'.format(n_clusters_))
    print('Cluster labels: {}'.format(set(labels)))
    print('Number of noise samples: {}'.format(n_noise_))

    # Add Labels (and its string version) to the dataframe
    df_piv_clean['Labels'] = labels

    print('df_piv_clean.head(5)')
    print(df_piv_clean.head(5))

    return df_piv_clean


def get_distance_matrix(df_in):
    if df_in is None or df_in.shape[0] == 0:
        return df_in

    df = df_in.copy()

    row_count = df.shape[0]
    distances = np.zeros((row_count, row_count))

    for idx1 in range(row_count-1):
        for idx2 in range(idx1+1, row_count):
            distances[idx1][idx2] = entropy(df.iloc[idx1]['KDE_vals'],
                                            df.iloc[idx2]['KDE_vals'])
            distances[idx2][idx1] = distances[idx1][idx2]

    df_out = pd.DataFrame(distances)
    df_out.fillna(0.00, inplace=True)
    distances_sum = df_out.apply(np.sum)
    argmin = distances_sum.argmin()
    return df_out, df.iloc[argmin]


def plot_clusters(df_in, folder):
    if df_in is None or df_in.shape[0] == 0:
        return df_in

    df = df_in.copy()

    num_labels = df['Labels'].nunique()
    print('num_labels: {}'.format(num_labels))

    labels = df['Labels'].unique()
    print('labels: {}'.format(labels))

    fig, axs = plt.subplots(num_labels, 2,
                            gridspec_kw={'hspace': 1.0, 'wspace': 0.5},
                            figsize=(15, 15))

    # Use num_labels - 1 in range, as we handle noise (-1) separately
    for label in labels:
        print('Label processed: {}'.format(label))

        # idx used in plot axes
        idx = 0
        if label != -1:
            idx = label
        else:
            idx = num_labels - 1

        df_lbl = df[df.Labels == label]

        distances, cluster_center = get_distance_matrix(df_lbl)
        print('Cluster center for label ' + str(label))
        print(cluster_center)

        # Histogram
        xh = cluster_center[0]
        axs[idx][0].hist(xh, density=True)
        axs[idx][0].title.set_text('Cluster ' + str(label) +
                                   ' (size ' + str(df_lbl.shape[0]) +
                                   ') AF histogram')

        # KDE
        xk = x_idx
        yk = cluster_center[1]
        axs[idx][1].plot(xk, yk)
        axs[idx][1].title.set_text('Cluster ' + str(label) +
                                   ' (size ' + str(df_lbl.shape[0]) +
                                   ') AF density estimate')

    plt.savefig(folder + '/dbscan_' + str(num_labels) + '.png')


def get_cluster_samples(df_in, sep, folder):
    if df_in is None or df_in.shape[0] == 0:
        return df_in

    df = df_in.copy()

    labels = df['Labels'].unique()
    print('labels: {}'.format(labels))

    for label in labels:
        df_lbl = df[df.Labels == label]
        df_lbl.to_csv(folder + '/cluster_' + str(label) + '.tsv', sep='\t')

# Run DBSCAN clustering algorithm on batch dataset
#
# Using 0.0075 for esp (epsilon) to yield ? clusters
# Using default value of 5 for min_samples
# Using get_kl_div method for metric. get_kl_div() calculates Kullback-Leibler
#   divergence, which measures the distance between 2 proabaility distributions
# Using None for metric_params, as the metric has no parameters
#

# 1. Cleaned the data (Removed rows with AF > 1.0)
# 2. Pivoted the data so all AFs of a batch are listed on one line
# 3. Calculated Kernel Density Estimates (KDE) of AFs of each batch
#      Evaluated them on 100 data points in range of 0.0 to 1.0
# 4. Ran DBSCAN clustering algorithm
#      epsilon: 0.0075
#      Used KL divergence to calculate distance between density estimate
#      metric: 'precomputed'. See note below
# 5. DBSCAN produced ? clusters
#      Data points not assigned to any cluster marked as Noise (or cluster -1)
# 6. For each cluster, found a representative batch
#      Calculated KL div. between every pair of batches in a cluster
#      Selected batch with the smallest sum of distances


# Calculated the distance matrix. We run the code below just once, and save the
# distance matrix to file. We pass the distance matrix file to DBSCAN. That way
# if we modify DBSCAN parameters (say, eps or num_sample), we avoid calculating
# the distance matrix repeatedly. Must set metric to 'precomputed'

def calculate_distance_matrix(input_file='gx-surveillance.json',
                              data_file='gx-all_variants.tsv',
                              path='/content/gdrive/MyDrive/Colab Notebooks/Clustering/lineage_overlap_data/',
                              study_accessions_file='Estonia_study_accessions.csv',
                              start_date='05-15-2021',
                              end_date='07-30-2021'):

    samples_df = get_samples(input_file=path+input_file,
                             study_accessions_file=path+study_accessions_file,
                             start_date=start_date,
                             end_date=end_date)

    print(samples_df.head())

    df = preprocess(path+data_file,
                    sep='\t',
                    samples_df=samples_df)
    distances, _ = get_distance_matrix(df)
    distances.to_csv(path+'distances_'+data_file, sep='\t')


def dbscan_clustering_wrapper(eps=0.0085,
                              min_samples=7,
                              path='/content/gdrive/MyDrive/Colab Notebooks/Clustering/',
                              data_file='gx-all_variants.tsv',
                              sep='\t',
                              data_folder='lineage_overlap_data',
                              results_folder='lineage_overlap_results',
                              metric='precomputed',
                              n_jobs=1,
                              input_file='gx-surveillance.json',
                              study_accessions_file='Estonia_study_accessions.csv',
                              start_date='05-15-2021',
                              end_date='07-30-2021'):

    folder = str(min_samples) + '_' + str(eps)
    print('folder: {}'.format(folder))
    full_path = os.path.join(path, results_folder, folder)
    print('full_path: {}'.format(full_path))
    os.mkdir(full_path)

    full_data_folder = os.path.join(path, data_folder)
    full_results_folder = os.path.join(path, results_folder, folder)

    df = dbscan_clustering(file_name=full_data_folder+'/'+data_file,
                           sep=sep,
                           eps=eps,
                           min_samples=min_samples,
                           metric=metric,
                           metric_params=None,
                           distances_file_name=full_data_folder+'/distances_'+data_file,
                           n_jobs=n_jobs,
                           input_file=full_data_folder+'/'+input_file,
                           study_accessions_file=full_data_folder+'/'+study_accessions_file,
                           start_date=start_date,
                           end_date=end_date)

    df.to_csv(full_results_folder + '/all_clusters_eps_' + str(eps) +
              '_min_samples_' + str(min_samples) + '.tsv', sep=sep)
    plot_clusters(df, folder=full_results_folder)
    get_cluster_samples(df_in=df, sep=sep, folder=full_results_folder)


if __name__ == '__main__':
    #
    # Run DBSCAN algorithm on 'data_file' in 'data_folder' and
    # save the results to 'results_folder'.
    #
    # Anup suggestion: Use scikit-learn grid search. That way
    # each parameter combination can be run in parallel.
    #
    # for min_samples in [2, 3, 4, 5, 6, 7]:
    #    for eps in [0.0200, 0.0225, 0.0250, 0.0275, 0.0300]:
    for min_samples in [2]:
        for eps in [0.0200]:
            print('\n\n\nRunning DBSCAN for min_samples: {}, eps: {}'.format(
                  min_samples, eps))
            dbscan_clustering_wrapper(eps=eps,
                                      min_samples=min_samples,
                                      path='/Users/kxk302/workspace/Covid_Clustering/',
                                      data_file='gx-all_variants.tsv',
                                      sep='\t',
                                      data_folder='lineage_overlap_data',
                                      results_folder='lineage_overlap_results',
                                      metric='precomputed',
                                      n_jobs=1,
                                      input_file='gx-surveillance.json',
                                      study_accessions_file='Estonia_study_accessions.csv',
                                      start_date='2020-09-15',
                                      end_date='2020-12-15')

    '''
    calculate_distance_matrix(input_file='gx-surveillance.json',
                              data_file='gx-all_variants.tsv',
                              path='lineage_overlap_data/',
                              study_accessions_file='Estonia_study_accessions.csv',
                              start_date='2020-09-15',
                              end_date='2020-12-15')
    '''
    '''
    distances = pd.read_csv('/Users/kxk302/workspace/Covid_Clustering/lineage_overlap_data/distances_gx-all_variants.tsv', 
                            sep='\t', index_col=0)

    distances.replace([np.inf], max_distance, inplace=True)
    distances.replace([np.nan], min_distance, inplace=True)
    distances[distances < 0] = 0.00

    num_rows = distances.shape[0]
    num_cols = distances.shape[1]

    for row in range(num_rows):
        for col in range(num_cols):
            if distances.iloc[row, col] < 0.00:
                print('value at row {} ans col {} is neagative: {}'.format(row, col, distances.iloc[row, col]))
    '''