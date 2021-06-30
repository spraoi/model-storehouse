import boto3
import joblib
import pandas as pd
import numpy as np
import tempfile

def get_bucket_and_key_from_s3_uri(uri):
    bucket, key = uri.split('/',2)[-1].split('/',1)
    return bucket, key

def download_obj_from_s3(bucket_name, key):
    bucket = boto3.resource("s3").Bucket(bucket_name)
    with tempfile.NamedTemporaryFile() as fp:
        bucket.download_fileobj(key, fp)
        loaded_model = joblib.load(fp.name)
    return loaded_model

def magnificent_map(df, remap_columns):
    PD_CAT = [
        "UNKNOWN",
        "ILL-DEFINED CONDITIONS",
        "CONGENITAL ANOMALIES",
        "DISEASES OF THE BLOOD",
        "SKIN & SUBCUTANEOUS TISSUE",
        "RESPIRATORY SYSTEM",
        "NERVOUS SYSTEM & SENSE ORGANS",
        "MENTAL & NERVOUS DISORDERS",
    ]
    SIC_HIGH = ['7371', '6321', '3548', '3444', '2731', '7514', '5063', '2297', '3751', '8231', '9531', '3586', '3823', '3672', '3271', '2865', '2399', '7819', '2841', '4011', '3561', '3363', '3629', '8999', '3366']
    SIC_MED = ['3841', '6371', '3549', '5722', '8222', '4522', '4111', '8631', '5122', '2022', '1623', '5141', '5641', '3496', '3731', '3674', '6411', '6153', '2821', '1521', '5944', '3545', '8621', '5099', '5015', '9224', '5072', '8042', '5311']
    COV_CORE = ["STD", "STDCORE", "STDBU", "STD1"]
    COV_VOL = ["STDVOL", "VS1", "VS2"]

    for col in remap_columns:

        if col == "Primary Diagnosis Category":
            df.loc[:, col] = df.loc[:, col].str.upper()
            df.loc[df[col] == "", col] = "UNKNOWN"
            df.loc[df[col].isnull(), col] = "UNKNOWN"
            df.loc[
                df[col].str.startswith("MENTAL, BEHAVIORAL & NEURO"), col
            ] = "MENTAL & NERVOUS DISORDERS"
            df.loc[
                df[col].str.startswith("DISEASES OF THE NERVOUS SYSTEM"), col
            ] = "NERVOUS SYSTEM & SENSE ORGANS"
            df.loc[
                df[col].str.startswith("DISEASES OF THE RESPIRATORY"), col
            ] = "RESPIRATORY SYSTEM"
            df.loc[
                df[col].str.startswith("DISEASES OF THE SKIN & SUBCUTANEOUS"), col
            ] = "SKIN & SUBCUTANEOUS TISSUE"
            df.loc[
                df[col].str.startswith("DISEASES OF THE BLOOD & BLOOD-FORMING"), col
            ] = "DISEASES OF THE BLOOD"
            df.loc[
                df[col].str.startswith("CONGENITAL MALFORMATIONS, DEFORMATIONS"), col
            ] = "CONGENITAL ANOMALIES"
            df.loc[~df[col].isin(PD_CAT), col] = "OTHERS"
        elif col == "SIC Code":
            #             df.loc[:, col] = pd.to_numeric(df.loc[:, col]).fillna(-99).astype(
            #                 int)  #to avoid float strings!
            df.loc[:, "SIC_risk_ind"] = (
                df.loc[:, col].replace(SIC_HIGH, "High").replace(SIC_MED, "Medium")
            )
            df.loc[~df["SIC_risk_ind"].isin(["High", "Medium"]), "SIC_risk_ind"] = "Low"

        elif col == "Coverage Code":
            df.loc[:, col] = (
                df.loc[:, col].replace(COV_CORE, "STDCORE").replace(COV_VOL, "STDVOL")
            )
            df.loc[~df[col].isin(["STDCORE", "STDVOL"]), col] = "OTHERS"

    return df


def resolve_formatting(df, date_cols, numeric_cols):
    """
    function to ensure correct date and numeric formatting
    """
    for col in list(df.columns):
        if col in date_cols:
            try:
                df.loc[:, col] = pd.to_datetime(df.loc[:, col])
            except:
                df.loc[:, col] = pd.to_datetime(df.loc[:, col], errors="coerce")
        elif col in numeric_cols:
            df.loc[:, col] = pd.to_numeric(df.loc[:, col])

    return df


def get_date_diff(col1, col2, interval):

    """
    difference between dates specified by the interval in ['Y','D','M']
    col1 and col2 are date colummns and col2 > col1
    """
    return (col2 - col1) / np.timedelta64(1, interval)


def add_policy_tenure_to_df(df):
    """
    returns a df with policy tenure column appended
    """
    df["policy_tenure"] = get_date_diff(
        df["Policy Effective Date"], df["Policy Termination Date"], interval="D"
    )
    df["policy_tenure_2"] = get_date_diff(
        df["Policy Effective Date"], df["Received Date"], interval="D"
    )
    df.loc[df["policy_tenure"].isnull(), "policy_tenure"] = df.loc[
        df["policy_tenure"].isnull(), "policy_tenure_2"
    ]
    df.drop("policy_tenure_2", axis=1, inplace=True)
    df.loc[df["policy_tenure"] <= 0, "policy_tenure"] = np.nan

    return df


def add_days_rep_to_df(df):
    """
    returns a df with days to report column appeneded
    """

    df["days_to_report"] = get_date_diff(
        df["Loss Date"], df["Received Date"], interval="D"
    )
    return df


def add_emp_tenure_to_df(df):
    """
    returns a df with employment tenure column appeneded
    """
    df["emp_tenure"] = get_date_diff(
        df["Insured Hire Date"], df["Loss Date"], interval="D"
    )
    col = "emp_tenure"
    df.loc[(df[col].isnull()) | (df[col] <= 0), col] = np.nan
    #     print(f"col {col}: missing= {df[col].isnull().sum()}")

    return df


def add_prognosis_days_to_df(df):
    """
    returns a df with prognosis days column appeneded
    """

    df["prognosis_days"] = get_date_diff(
        df["Loss Date"], df["Duration Date"], interval="D"
    )
    col = "prognosis_days"

    df.loc[(df[col].isnull()) | (df[col] <= 0), col] = (
        pd.to_numeric(df.loc[(df[col].isnull()) | (df[col] <= 0), "Duration Months"])
        * 30
    )
    df.loc[df[col] <= 0, col] = np.nan
    return df


def to_category(df, cat_cols):
    """
    return df with the categorical features formatted as pandas category dtype
    """
    for col in cat_cols:
        df.loc[:, col] = df.loc[:, col].astype("category")
    return df


def test_train_match(train_template, test_data):
    missing_cols = set(train_template.columns) - set(test_data.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test_data[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test_data = test_data[train_template.columns].copy()
    return test_data


def get_na_rows(test_data):
    na_inds = []
    for i in range(test_data.shape[0]):
        if test_data.iloc[i, :].isnull().any():
            na_inds.append("Y")
        else:
            na_inds.append("N")
    test_data.loc[:, "NA_row"] = na_inds
    return test_data


def posterior_correction(p1_orig, p_1_train, pred):

    cf = p1_orig / p_1_train
    ncf = (1 - p1_orig) / (1 - p_1_train)
    pcorr = (pred * cf) / (pred * cf + (1 - pred) * ncf)
    return pcorr


