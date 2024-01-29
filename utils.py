import pandas as pd 
import numpy as np

def remove_duplicate_annot(dframe: pd.DataFrame, col_name: list) -> pd.DataFrame:
    '''Drop duplicates in annotations of chromosome locations.'''
    dframe.drop_duplicates(subset=col_name, keep='first', inplace=True, ignore_index=True)
    return dframe

def add_annotations(dprofiles: pd.DataFrame, dannot: pd.DataFrame) -> pd.DataFrame:
    '''Add annotations for genes'''
    df_merged = dprofiles.merge(dannot,
                                on='Metadata_JCP2022',
                                how='left')
    return df_merged

def split_arm(locus: str) -> str:
    if pd.isna(locus):
        return locus
    
    elif 'p' in locus:
        return locus.split('p')[0] + 'p'
    
    elif 'q'in locus:
        return locus.split('q')[0] + 'q'
    
def process_merged_dframe(dframe: pd.DataFrame) -> pd.DataFrame:
    '''Clean up chromosome annotations.'''
    dframe.drop(np.where(dframe.Chromosome=='reserved')[0], inplace=True)
    dframe['Chromosome'] = dframe['Chromosome'].apply(lambda x: '12' 
                                                      if x=='12 alternate reference locus'
                                                      else x)
    
    dframe = dframe.rename(columns={'Approved_symbol': 'Metadata_approved_symbol',
                                    'Locus': 'Metadata_locus',
                                    'Chromosome': 'Metadata_chromosome'})
    dframe['Metadata_arm'] = dframe['Metadata_locus'].apply(lambda x: split_arm(x))
    return dframe


def correction(dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract the mean of each feature per chromosome arm.

    Parameters
    ----------
    dframe : pandas.DataFrame
        Dataframe with features and metadata.

    Returns
    -------
    pandas.DataFrame
        Dataframe with features and metadata, with each feature subtracted by the mean of that feature per chromosome arm.
    """
    feature_cols = dframe.filter(regex="^(?!Metadata_)").columns
    control = dframe[pd.isna(dframe['Metadata_arm'])].reset_index(drop=True)
    non_control = dframe[~pd.isna(dframe['Metadata_arm'])].reset_index(drop=True)
    non_control[feature_cols] = non_control.groupby("Metadata_arm")[feature_cols].transform(
        lambda x: x - x.mean()
    )
    dcorrected = pd.concat([control, non_control], ignore_index=True)
    return dcorrected

def workflow(dprofiles: pd.DataFrame, dchrom: pd.DataFrame) -> pd.DataFrame:
    '''Merge the profiles and chromosome annotations.'''
    
    # Remove duplicated annotations
    dchrom = remove_duplicate_annot(dchrom, ['Approved_symbol'])

    # Merge profiles and chromosome annotations
    df_merged = dprofiles.merge(dchrom, 
                         how='left', 
                         left_on='Metadata_Symbol', 
                         right_on='Approved_symbol').reset_index(drop=True)
    
    # Clean up chromosome annotations
    df_merged = process_merged_dframe(df_merged)

    # Chromosome arm correction
    df_merged = correction(df_merged)

    return df_merged

