import numpy as np
import pandas as pd
import boto3
import joblib
import tempfile
from typing import List


def tokenize_pd_code(df):

    df[['pd_code_1', 'pd_code_2']] = pd.DataFrame(
        df['PrimaryDiagnosisCode'].fillna("na.na").str.split(".", expand=True))
    return df.drop('PrimaryDiagnosisCode', axis=1)

def get_date_diff(col1, col2, interval):
    """difference between dates specified by the interval in ['Y','D','M']
    col1 and col2 are date colummns and col2 > col1"""

    return (col2 - col1) / np.timedelta64(1, interval)

def add_emp_tenure_to_df(df):
    """
    returns a df with employment tenure column appeneded
    """
    df["emp_tenure"] = get_date_diff(
        df["InsuredHireDate"], df["LossDate"], interval="D"
    )
    
    return df.drop("InsuredHireDate", axis=1)


class CategoricalGrouper:

    """we are fitting the categorical grouper on a reference data
    using min_freq, which means all the low-frequency categories are
    pushed into a garbage or OOV category"""

    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.is_fitted = False


    def fit(self, X, categorical: List):
        _non_oov_dict = dict()
        for col in categorical:
            _df = pd.DataFrame(X[col].value_counts()).reset_index(drop=False)
            _df.columns = ["index", "counts"]
            non_oov = list(_df.loc[_df['counts'] >= self.min_freq, "index"])
            _non_oov_dict[col] = non_oov
#             X.loc[~X[col].isin(non_oov), col] = 'OOV'
        self._non_oov_dict = _non_oov_dict
        self.is_fitted = True
        return None


    def transform(self, X, categorical: List):
        if self.is_fitted:
            for col in categorical:
                non_oov = self._non_oov_dict[col]
                X.loc[~X[col].isin(non_oov), col] = 'OOV'

            return X
        else:
            raise ValueError(
                "The categorical grouper needs to be fit before calling .transform()"
            )

def download_obj_from_s3(bucket_name, key):
    bucket = boto3.resource("s3").Bucket(bucket_name)
    print(bucket_name, key)
    with tempfile.NamedTemporaryFile() as fp:
        bucket.download_fileobj(key, fp)
        loaded_model = joblib.load(fp.name)
    return loaded_model


# download_obj_from_s3("spr-ml-artifacts","dev/MLMR_Bridging/artifacts/all_artifacts_10-05-2021.joblib")