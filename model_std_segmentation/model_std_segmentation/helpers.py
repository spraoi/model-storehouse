import functools
import json
import logging
import tempfile

import boto3
import joblib
import numpy as np
import pandas as pd

numeric_cols = ["Insured Annualized Salary", "Policy Lives", "Insured Age at Loss"]

date_cols = [
    "Loss Date",
    "Policy Effective Date",
    "Policy Termination Date",
    "Approval Date",
    "Closed Date",
    "Last Payment To Date",
    "First Payment From Date",
    "Nurse Cert End Date",
    "Insured Hire Date",
    "Received Date",
]

categorical_cols = [
    "Claim State",
    "Primary Diagnosis Category",
    "SIC Category",
    "Insured Gender",
    "Occ Category",
    "Claim Cause Desc",
]

date_diff_cols = list(set(date_cols).difference({"Loss Date", "Nurse Cert End Date"}))


def get_bucket_and_key_from_s3_uri(uri: str):
    bucket, key = uri.split("/", 2)[-1].split("/", 1)
    return bucket, key


def download_model_from_s3(bucket_name, key):
    bucket = boto3.resource("s3").Bucket(bucket_name)
    with tempfile.NamedTemporaryFile() as fp:
        bucket.download_fileobj(key, fp)
        loaded_model = joblib.load(fp.name)
    return loaded_model


def _check_for_no_data(df: pd.DataFrame, hint: str = None) -> int:
    if df.empty:
        logging.info(f"no data post {hint}")
        return 1
    else:
        return 0


def _payment_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[~(df["Last Payment To Date"] < df["First Payment From Date"])]


def _fill_date_cols(df: pd.DataFrame, date_cols: list = date_diff_cols) -> pd.DataFrame:
    for col in date_cols:
        df[col].fillna(pd.to_datetime("01/01/1997"), inplace=True)
    return df


def _resolve_formatting(
    df: pd.DataFrame, date_cols: list = date_cols, numeric_cols: list = numeric_cols
) -> pd.DataFrame:
    for col in list(df.columns):
        if col in date_cols:
            try:
                df.loc[:, col] = pd.to_datetime(df.loc[:, col])
            except:
                df.loc[:, col] = pd.to_datetime(df.loc[:, col], errors="coerce")
        elif col in numeric_cols:
            df.loc[:, col] = pd.to_numeric(df.loc[:, col])

    return df


def _to_category(df: pd.DataFrame, cat_cols: list = categorical_cols) -> pd.DataFrame:
    for col in cat_cols:
        df.loc[:, col] = df.loc[:, col].astype("category")
    return df


def get_date_diff(col1: pd.Series, col2: pd.Series, interval: str = "D") -> pd.Series:
    diff = col2 - col1
    diff /= np.timedelta64(1, interval)

    return diff


def _fix_nurse_date(df: pd.DataFrame) -> pd.DataFrame:
    df["Nurse Cert End Date"][df["Nurse Cert End Date"].isna()] = 0
    df["Nurse Cert End Date"][df["Nurse Cert End Date"].notna()] = 1
    df["Nurse Cert End Date"] = df["Nurse Cert End Date"].astype(int)
    return df


def _filter_bank_two(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Loss Date"] > df["Policy Effective Date"]]
    df = df[~(df["Loss Date"] > df["Policy Termination Date"])]
    df = df[~(df["Loss Date"] > df["Approval Date"])]
    df = df[~(df["Loss Date"] > df["Closed Date"])]

    return df


def _filter_bank_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["Claim Status Code"].isin(["92"])]
    df = _resolve_formatting(df, date_cols=date_cols, numeric_cols=numeric_cols)
    df = df[df["Loss Date"].notna()]
    df = df[df["Primary Diagnosis Code"].notna()]
    df = df.dropna(subset=numeric_cols, how="any")
    df = df[(df["Insured Age at Loss"] > 16.0) & (df["Insured Age at Loss"] < 90.0)]
    df = df[df["Policy Lives"].notna()]
    df["Policy Lives"] = df["Policy Lives"].astype(int)
    df = df[~df["Policy Lives"] <= 0]
    df = df[~df["Claim State"].isna()]
    df = df[~df["Insured Gender"].isna()]

    return df


def _fill_unknown(df: pd.DataFrame) -> pd.DataFrame:
    df["Occ Category"] = df["Occ Category"].fillna("unknown")
    df["Claim Cause Desc"] = df["Claim Cause Desc"].fillna("UNKNOWN")
    df["Primary Diagnosis Category"] = df["Primary Diagnosis Category"].fillna(
        "UNKNOWN"
    )

    return df


def _check_na_counts_thresh(df: pd.DataFrame, thresh: int = 5) -> int:
    test_na_sum_date = (
        df[list(set(date_cols).difference({"Loss Date", "Nurse Cert End Date"}))]
        .isnull()
        .any()
        .sum()
    )
    test_na_sum_cat = df[categorical_cols].isnull().any().sum()
    if test_na_sum_date + test_na_sum_cat > thresh:
        logging.error("too many NaN values while comparing with threshold")
        logging.info(test_na_sum_date + test_na_sum_cat)
        df["bad_data"] = 1
        return df
    mandatory_date_cols = [
        "Policy Termination Date",
        "Policy Effective Date",
        "Loss Date",
        "Approval Date",
        "Closed Date",
    ]
    df = df.dropna(subset=mandatory_date_cols)
    if _check_for_no_data(df):
        logging.error("essential date cols null")
        df["bad_data"] = 1
    df["bad_data"] = 0
    return df


def _remap_features(df: pd.DataFrame) -> pd.DataFrame:

    pd_cat = [
        "UNKNOWN",
        "ILL-DEFINED CONDITIONS",
        "CONGENITAL ANOMALIES",
        "DISEASES OF THE BLOOD",
        "SKIN & SUBCUTANEOUS TISSUE",
        "RESPIRATORY SYSTEM",
        "NERVOUS SYSTEM & SENSE ORGANS",
        "MENTAL & NERVOUS DISORDERS",
    ]

    col = "Primary Diagnosis Category"
    df.loc[(~df[col].isin(pd_cat) & df[col].notnull()), col] = "OTHERS"
    return df


def _date_diff_all(df: pd.DataFrame, date_cols: list = date_diff_cols) -> pd.DataFrame:
    for col in date_cols:
        df.loc[:, col] = get_date_diff(df.loc[:, "Loss Date"], df.loc[:, col], "D")
        df[col] = df[col].astype(int)
    return df


def _replace_state(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df.loc[:, "Claim State"] = df["Claim State"].map(mapping)
    df.loc[:, "Claim State"] = df.loc[:, "Claim State"].astype("category")
    return df


def _compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def _compose(*fs):
    return functools.reduce(_compose2, fs)


def _generate_payload(df):
    payload = df.loc[
        :,
        [
            "Claim Identifier",
            "predictedSegment",
            "predictedProbability",
            "tier",
            "tierHint",
        ],
    ].copy()
    payload.columns = [
        "claimNumber",
        "predictedSegment",
        "predictedProbability",
        "tier",
        "tierHint",
    ]
    payload_json = json.loads(payload.to_json(orient="records"))
    predicted_claim = payload_json[0] if payload_json else None
    return [
        {
            "inputDataSource": f"{predicted_claim.get('claimNumber')}:0",
            "entityId": predicted_claim.get("claimNumber"),
            "predictedResult": predicted_claim,
        }
    ]
