import boto3
import joblib
import pandas as pd
import numpy as np
import tempfile


def get_bucket_and_key_from_s3_uri(uri):
    bucket, key = uri.split("/", 2)[-1].split("/", 1)
    return bucket, key


def download_obj_from_s3(bucket_name, key):
    bucket = boto3.resource("s3").Bucket(bucket_name)
    print(bucket_name, key)
    with tempfile.NamedTemporaryFile() as fp:
        bucket.download_fileobj(key, fp)
        loaded_model = joblib.load(fp.name)
    return loaded_model


def magnificent_map(df, remap_columns):
    PRE_EX_MAP = {"YES": "Y", "NO": "N"}
    SS_PURSUE_MAP = {"YES": "Y", "NO": "N", "Yes": "Y", "No": "N"}
    SAL_FREQ_MAP = {
        "A": "YEARLY",
        "M": "MONTHLY",
        "W": "WEEKLY",
        "B": "BI-WEEKLY",
        "H": "HOURLY",
    }

    for col in remap_columns:
        if col == "Insured Salary Ind":
            df[col].replace(SAL_FREQ_MAP, inplace=True)
        elif col == "Pre-Ex Outcome":
            df[col].replace(PRE_EX_MAP, inplace=True)
        elif col == "SS Pursue Ind":
            df[col].fillna("NA").replace(SS_PURSUE_MAP, inplace=True)
    return df


def resolve_formatting(df, date_cols, numeric_cols):
    """
    function to ensure correct date and numeric formatting
    """
    for col in list(df.columns):
        if col in date_cols:
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


def add_payment_lag_to_df(df):
    """
    return a df with last first payment lag column appended
    """
    df["last_first_payment_lag"] = df.apply(
        lambda x: get_date_diff(
            x["First Payment From Date"], x["Last Payment To Date"], interval="D"
        ),
        axis=1,
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
    columns = [
        "Insured Age at Loss",
        "Insured Annualized Salary",
        "Policy Lives",
        "days_to_report",
        "emp_tenure",
        "prognosis_days",
        "Insured Gender",
        "Insured Salary Ind",
        "Primary Diagnosis Category",
        "Coverage Code",
        "last_first_payment_lag",
    ]
    col_locs = [test_data.columns.get_loc(col) for col in columns]
    na_inds = []
    for _, row in test_data.iterrows():
        na_inds.append((row[col_locs].isnull() | row[col_locs].isna()).any())

    test_data.loc[:, "NA_row"] = ["Y" if x else "N" for x in na_inds]
    return test_data


def posterior_correction(p1_orig, p_1_train, pred):

    cf = p1_orig / p_1_train
    ncf = (1 - p1_orig) / (1 - p_1_train)
    pcorr = (pred * cf) / (pred * cf + (1 - pred) * ncf)
    return pcorr


# another alternative way to correct probabilities
# >99.4% correlation exists b/w both methods and hence are equivalent
# source: https://andrewpwheeler.com/2020/07/04/adjusting-predicted-probabilities-for-sampling/
def calibrate_predictions(predictions, p1_train, p1_original):
    a = predictions / (p1_train / p1_original)
    comp_cond = 1 - predictions
    comp_train = 1 - p1_train
    comp_original = 1 - p1_original
    b = comp_cond / (comp_train / comp_original)
    return a / (a + b)
