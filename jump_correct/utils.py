import pandas as pd
import numpy as np
import anndata as ad
import cupy as cp
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

def get_meta_cols(df):
    """return a list of metadata columns"""
    return df.filter(regex="^(Metadata_)").columns


def get_feature_cols(df):
    """returna  list of featuredata columns"""
    return df.filter(regex="^(?!Metadata_)").columns


def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_meta_cols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_feature_cols(df)]
    
def obsm_to_df(adata: ad.AnnData, obsm_key: str,
               columns: list[str] | None) -> pd.DataFrame:
    '''Convert AnnData object to DataFrame using obs and obsm properties'''
    meta = adata.obs.reset_index(drop=True)
    feats = adata.obsm[obsm_key]
    n_feats = feats.shape[1]
    if not columns:
        columns = [f'{obsm_key}_{i:04d}' for i in range(n_feats)]
    data = pd.DataFrame(feats, columns=columns)
    return pd.concat([meta, data], axis=1)
    
def rescale(df, range2):
    range1 = [df.min().min(), df.max().max()]
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (df - range1[0]) / delta1) + range2[0]

def treatment_level(df, feature_name="Metadata_JCP2022"):
    """
    Computes the treatment level consensus profiles.
    Parameters:
    -----------
    df: pandas.DataFrame
        dataframe of profiles
    feature_name: str
        Name of the treatment feature
    Returns:
    -------
    pandas.DataFrame of treatment aggregated profiles
    """
    feauture_cols = get_feature_cols(df)
    meta_cols = get_meta_cols(df)
    
    df_features = df.groupby([feature_name])[feauture_cols].mean()
    df_features[feature_name] = df_features.index
    df_features.reset_index(drop=True, inplace=True)
    
    df_treatment = get_metadata(df).drop_duplicates(subset=[feature_name])
    df_treatment = df_treatment.merge(df_features,
                                     on=feature_name,
                                     how='inner').reset_index(drop=True)
    return df_treatment
    
def remove_nan_features(df):
    r, c = np.where(df.isna())
    features_to_remove = [
        _
        for _ in list(df.columns[list(set(c))])
        if not _.startswith("Metadata_")
    ]
    print(f"Removed nan features: {features_to_remove}")
    return df.drop(features_to_remove, axis=1)
    
def annotate_gene(df, df_meta):
    if 'Metadata_Symbol' not in df.columns:
        df = df.merge(df_meta[['Metadata_JCP2022', 'Metadata_Symbol']], 
                        on='Metadata_JCP2022', 
                        how='inner')
    return df

def annotate_chromosome(df, df_meta):
    if 'Metadata_Locus' not in df.columns:
        df_meta_copy = df_meta.drop_duplicates(subset=['Approved_symbol']).reset_index(drop=True)
        df_meta_copy.columns = ['Metadata_' + col 
                           if not col.startswith('Metadata_') 
                           else col for col in df_meta_copy.columns ]
        
        df = df.merge(df_meta_copy, 
                     how='left', 
                     left_on='Metadata_Symbol', 
                     right_on='Metadata_Approved_symbol').reset_index(drop=True)
        
    if 'Metadata_arm' not in df.columns:
        df['Metadata_arm'] = df['Metadata_Locus'].apply(lambda x: split_arm(str(x)))
        
    return df

def map_chrom(x):
    chrom_dict = {'X': 23, 'Y': 24}
    
    try:
        result = int(x)
        return result
    except:
        try: return chrom_dict[x]
        except: 
            return np.nan

def split_arm(locus):
    if 'p' in locus:
        return locus.split('p')[0] + 'p'
    
    elif 'q'in locus:
        return locus.split('q')[0] + 'q'
        
def sort_on_chromosome(df, col_name='Metadata_Chromosome', locus_name='Metadata_Locus'):
    df[col_name] = df[col_name].apply(lambda x: '12' if x=='12 alternate reference locus' else x)
    df['Chromosome_num'] = df[col_name].apply(lambda x: map_chrom(x))
    df = df.dropna(subset=['Chromosome_num'])
    df.sort_values(by=['Chromosome_num', locus_name], ignore_index=True, inplace=True)
    return df.drop('Chromosome_num', axis=1)

def compute_group_border(df, col_name):
    groups = df.groupby(col_name).groups
    idx_list = []
    for group in groups:
        idx_list.append(groups[group][0])
    idx_list.sort()
    return idx_list

def transform_data(
    data: pd.DataFrame,
    variance=0.98,
) -> pd.DataFrame:
    """Transform data by scaling and applying PCA. Data is scaled by plate
    before and after PCA is applied. The experimental replicates are averaged
    together by taking the mean.

    Parameters
    ----------
    data : pd.DataFrame
        Data to transform
    metadata_cols : list, optional
        Metadata columns, by default CPG_METADATA_COLS
    variance : float, optional
        Variance to keep after PCA, by default 0.98

    Returns
    -------
    pd.DataFrame
        Transformed data
    """
    metadata = get_metadata(data)
    features = get_featuredata(data)
    
    for plate in metadata.Metadata_Plate.unique():
        scaler = StandardScaler()
        features.loc[metadata.Metadata_Plate == plate, :] = scaler.fit_transform(
            features.loc[metadata.Metadata_Plate == plate, :]
        )

    features = pd.DataFrame(PCA(variance).fit_transform(features))

    for plate in metadata.Metadata_Plate.unique():
        scaler = StandardScaler()
        features.loc[metadata.Metadata_Plate == plate, :] = scaler.fit_transform(
            features.loc[metadata.Metadata_Plate == plate, :]
        )

    return pd.concat([metadata, features], axis=1)


def order_dataframe_based_on_chromosome(df, df_gene, df_chrom):
    df = treatment_level(df, 'Metadata_JCP2022')
    df = remove_nan_features(df)
    df = annotate_gene(df, df_gene)
    df = annotate_chromosome(df, df_chrom)
    # df = transform_data(df) # Perform PCA
    # df = treatment_level(df, 'Metadata_JCP2022')
    df = sort_on_chromosome(df)
    return df
    
def get_cosine_similarity(treat_df, var_feat):
    """
    Computes cosine similarity of treatments.
    Parameters:
    -----------
    treat_df: pandas.DataFrame
        dataframe of treatment profiles
    var_feat: str
        Name of the treatment column
    Returns:
    -------
    pandas.DataFrame of shape (treatment*treatment, 3)
    """
    
    feature_cols = get_feature_cols(treat_df)
    COS = cosine_similarity(treat_df[feature_cols], treat_df[feature_cols])
    
    df = pd.DataFrame(data=COS, index=list(treat_df[var_feat]),
                          columns=list(treat_df[var_feat]))
    df = df.reset_index().melt(id_vars=["index"])
    return df

def combine_two_matrices(X_1, X_2):
    if not isinstance(X_1, cp.ndarray):
        X_1 = cp.asarray(X_1)
    if not isinstance(X_2, cp.ndarray):
        X_2 = cp.asarray(X_1)

    # Create an empty matrix with the same size as the cosine similarity matrices
    combined_matrix = cp.zeros_like(X_1)
    
    # Fill the upper triangle of the matrix with values from cosine_sim_A
    cp.fill_diagonal(combined_matrix, 1)  # Fill the diagonal with ones
    combined_matrix[cp.triu_indices_from(combined_matrix, k=1)] = X_1[cp.triu_indices_from(X_1, k=1)]
    
    # Fill the lower triangle of the matrix with values from cosine_sim_A_prime
    combined_matrix[cp.tril_indices_from(combined_matrix, k=-1)] = X_2[cp.tril_indices_from(X_2, k=-1)]
    return combined_matrix

def quantile_normalize_to_normal(COS, mean=0, std=0.2):
    # cp.fill_diagonal(COS, 0)    
    COS_cpu = cp.asnumpy(COS)
    
    COS_sorted = cp.sort(COS).reshape(-1)
    COS_sorted = cp.unique(COS_sorted)
    COS_sorted = cp.asnumpy(COS_sorted)
    
    original_shape = COS_cpu.shape
    COS_cpu = COS_cpu.reshape(-1)
    
    # Generate the target normal distribution quantiles
    target_quantiles = stats.norm.ppf(np.linspace(0.5/len(COS_sorted),1-0.5/len(COS_sorted),len(COS_sorted)), 
                                      loc=mean, 
                                      scale=std).astype('float16')
    
    # Create a mapping from the original sorted positions to the target quantiles
    quantile_mapping = dict(zip(COS_sorted, target_quantiles))
    
    # Map the original cosine similarities to the target quantiles
    norm_COS = np.array([quantile_mapping[val] for val in COS_cpu])

    norm_COS = cp.asarray(norm_COS)
    norm_COS = norm_COS.reshape(original_shape)
    
    cp.fill_diagonal(norm_COS, 1)  # Fill the diagonal with ones

    del COS_cpu
    del COS_sorted
    
    return norm_COS
