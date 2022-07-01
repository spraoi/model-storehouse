def predict(**kwargs):
    import json
    import logging
    import sys
    import tempfile

    import boto3
    import joblib
    import numpy as np
    import pandas as pd

    pd.options.mode.chained_assignment = None

    def get_bucket_and_key_from_s3_uri(uri: str):
        bucket, key = uri.split("/", 2)[-1].split("/", 1)
        return bucket, key

    def download_model_from_s3(bucket_name, key):
        bucket = boto3.resource("s3").Bucket(bucket_name)
        with tempfile.NamedTemporaryFile() as fp:
            bucket.download_fileobj(key, fp)
            loaded_model = joblib.load(fp.name)
        return loaded_model

    def _check_for_no_data(df: pd.DataFrame, hint: str = None) -> None:
        if df.empty:
            logging.info(f"no data after {hint}")
            sys.exit()

    def _fill_date_cols(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
        for col in date_cols:
            df[col].fillna(pd.to_datetime("01/01/1997"), inplace=True)
        return df

    def _resolve_formatting(
        df: pd.DataFrame, date_cols: list, numeric_cols: list
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

    def _to_category(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        for col in cat_cols:
            df.loc[:, col] = df.loc[:, col].astype("category")
        return df

    def get_date_diff(
        col1: pd.Series, col2: pd.Series, interval: str = "D"
    ) -> pd.Series:

        diff = col2 - col1
        diff /= np.timedelta64(1, interval)

        return diff

    def _fix_nurse_date(df: pd.DataFrame) -> pd.DataFrame:
        df["Nurse Cert End Date"][df["Nurse Cert End Date"].isna()] = 0
        df["Nurse Cert End Date"][df["Nurse Cert End Date"].notna()] = 1
        df["Nurse Cert End Date"] = df["Nurse Cert End Date"].astype(int)
        return df

    def _filter_bank_two(df: pd.DataFrame) -> pd.DataFrame:
        if not df["Policy Effective Date"].isnull().bool():
            df = df[df["Loss Date"] > df["Policy Effective Date"]]

        if not df["Policy Termination Date"].isnull().bool():
            df = df[~(df["Loss Date"] > df["Policy Termination Date"])]
        if not df["Approval Date"].isnull().bool():
            df = df[~(df["Loss Date"] > df["Approval Date"])]
        if not df["Closed Date"].isnull().bool():
            df = df[~(df["Loss Date"] > df["Closed Date"])]

        df.reset_index(drop=True, inplace=True)
        return df

    for artifact in kwargs.get("artifact"):
        if artifact.get("dataName") == "combined_artifacts":
            model_bucket, model_key = get_bucket_and_key_from_s3_uri(
                artifact.get("dataValue")
            )

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

    scaler_cols = [
        "Policy Effective Date",
        "Policy Termination Date",
        "Approval Date",
        "Closed Date",
        "Last Payment To Date",
        "First Payment From Date",
        "Nurse Cert End Date",
        "Insured Hire Date",
        "Received Date",
        "Insured Annualized Salary",
        "Policy Lives",
        "Insured Age at Loss",
    ]

    def _filter_bank_one(df: pd.DataFrame) -> pd.DataFrame:
        df = df[~df["Claim Status Code"].isin(["92"])]
        df = _resolve_formatting(df, date_cols=date_cols, numeric_cols=numeric_cols)
        df = df[df["Loss Date"].notna()]
        df = df.dropna(subset=numeric_cols, how="any")
        df = df[(df["Insured Age at Loss"] > 16.0) & (df["Insured Age at Loss"] < 90.0)]

        return df

    def _filter_bank_three(df: pd.DataFrame) -> pd.DataFrame:
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

    def _check_na_counts_thresh(df: pd.DataFrame, thresh: int = 5) -> pd.DataFrame:
        test_na_sum_date = (
            df[list(set(date_cols).difference({"Loss Date", "Nurse Cert End Date"}))]
            .isnull()
            .any()
            .sum()
        )
        test_na_sum_cat = df[categorical_cols].isnull().any().sum()
        if test_na_sum_date + test_na_sum_cat > thresh:
            logging.info("too many NaN values while comparing with threshold")
            logging.info(test_na_sum_date + test_na_sum_cat)
            sys.exit()
        else:
            return df

    def _remap_features(df: pd.DataFrame, remap_columns: list) -> pd.DataFrame:

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

        for col in remap_columns:
            if col == "Primary Diagnosis Category":
                df.loc[(~df[col].isin(pd_cat) & df[col].notnull()), col] = "OTHERS"
        return df

    def _date_diff_all(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
        for col in date_cols:
            df.loc[:, col] = get_date_diff(df.loc[:, "Loss Date"], df.loc[:, col], "D")
            df[col] = df[col].astype(int)
        return df

    def _replace_state(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        df.loc[:, "Claim State"] = df["Claim State"].map(mapping)
        df.loc[:, "Claim State"] = df.loc[:, "Claim State"].astype("category")
        return df

    # artifacts = download_model_from_s3(model_bucket, model_key)
    with open("./data/combined_artifacts_120k.sav", "rb") as f:
        artifacts = joblib.load(f)
    robust_scaler_obj, catboost_enc_obj, rf_model, template_obj, remap_obj = artifacts

    input_data = pd.DataFrame([kwargs.get("inputs").get("claim")])
    input_data = _filter_bank_one(input_data)
    _check_for_no_data(input_data, "filter bank 1")

    input_data = input_data[date_cols + numeric_cols + categorical_cols]
    input_data = _check_na_counts_thresh(input_data)
    input_data = _filter_bank_two(input_data)
    _check_for_no_data(input_data, "filter bank 2")

    input_data = _filter_bank_three(input_data)
    _check_for_no_data(input_data, "filter bank 3")

    input_data = _fill_date_cols(
        input_data,
        date_cols=list(set(date_cols).difference({"Loss Date", "Nurse Cert End Date"})),
    )
    input_data = input_data[
        ~(input_data["Last Payment To Date"] < input_data["First Payment From Date"])
    ]
    input_data = _fill_unknown(input_data)
    input_data = _remap_features(
        input_data, remap_columns=["Primary Diagnosis Category"]
    )
    input_data = _fix_nurse_date(input_data)
    input_data = _date_diff_all(
        input_data,
        date_cols=list(set(date_cols).difference({"Loss Date", "Nurse Cert End Date"})),
    )

    input_data = _to_category(input_data, cat_cols=categorical_cols)
    input_data.drop(columns=["Loss Date"], inplace=True)
    df = _replace_state(input_data, remap_obj)
    df.reset_index(inplace=True, drop=True)

    df.loc[:, scaler_cols] = robust_scaler_obj.transform(
        df.loc[:, scaler_cols].to_numpy()
    )

    df_enc = catboost_enc_obj.transform(df[categorical_cols])
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, df_enc], axis=1)

    prediction = rf_model.predict(df)
    class_confidence = float(rf_model.predict_proba(df)[0][prediction])
    df["prediction"] = prediction
    df["class_confidence"] = class_confidence

    # payload_data = test_final.loc[
    #     :, ["Claim Identifier", "p_corrected", "p_labels_corrected"]
    # ].copy()  # ,'Claim Status Category'
    # payload_data.columns = [
    #     "claimNumber",
    #     "predictedProbability",
    #     "predictedValue",
    # ]  # ,'claimStatusCategory'
    # prediction_json = json.loads(payload_data.to_json(orient="records"))
    # predicted_claim = prediction_json[0] if prediction_json else None
    # return [{"inputDataSource":f"{predicted_claim.get('claimNumber')}:0","entityId":predicted_claim.get("claimNumber"),"predictedResult":predicted_claim}]
    return "done"


# example input

print(
    predict(
        model_name="model_std_segmentation",
        artifact=[
            {
                "dataId": "55dcc659-d0c5-42aa-b9bf-a0325a2997b9",
                "dataName": "combined_artifacts",
                "dataType": "artifact",
                "dataValue": "s3://spr-ml-artifacts/prod/segmentation/combined_artifacts_120k.sav",
                "dataValueType": "str",
            }
        ],
        inputs={
            "claim": {
                "Mental Nervous Ind": None,
                "Recovery Amt": 0.0,
                "Modified RTW Date": None,
                "Any Occ period": None,
                "__root_row_number": 366,
                "Claim Number": "GDC-72418",
                "Policy Effective Date": "01/01/2017",
                "DOT Exertion Level (Primary)": "Unknown",
                "Last Payment To Date": "01/28/2021",
                "DOT Desc": None,
                "Elimination Period": None,
                "DOT Code": None,
                "Voc Rehab Outcome": None,
                "Policy Line of Business": "STD",
                "Expected Term Date": None,
                "Clinical End Date": None,
                "Voc Rehab Service Requested": None,
                "Policy Termination Date": None,
                "Any Occ Start Date": None,
                "Duration Months": 1,
                "MentalNervousDesc": None,
                "Closed Date": None,
                "Insured State": "CA",
                "SS Pri Award Amt": None,
                "Any Occ Decision Due Date": None,
                "Elimination Days": 0,
                "SS Pursue Ind": None,
                "Any Occ Ind": None,
                "Elimination Ind": None,
                "__row_number": 366,
                "Plan Benefit Pct": 0.6,
                "Claim Status Description": "Benefit Case Under Review",
                "Secondary Diagnosis Code": "M54.17",
                "Secondary Diagnosis Category": None,
                "Claim Identifier": "GDC-72418-01",
                "SS Pri Award Eff Date": None,
                "Pre-Ex Outcome": "Y",
                "Claim Status Category": "ACTIVE",
                "Primary Diagnosis Code": "M51.27",
                "Voc Rehab Status": None,
                "Claim Cause Desc": "OTHER ACCIDENT",
                "Insured Salary Ind": "BI-WEEKLY",
                "Insured Zip": "93447",
                "SIC Code": 7349,
                "First Payment From Date": "01/28/2021",
                "SS Reject Code": None,
                "Any Occ Decision Made Date": None,
                "SIC Category": None,
                "Insured Age at Loss": 53,
                "Received Date": "02/12/2021",
                "Secondary Diagnosis Desc": "Radiculopathy, lumbosacral region",
                "Voc Rehab TSA Date": None,
                "TSA Ind": "N",
                "Secondary Diagnosis Category Desc": None,
                "SIC Desc": "BUILDING MAINTENANCE SERVICES, NEC",
                "Claim State": "CA",
                "ThirdPartyReferralDescription": None,
                "Occ Code": None,
                "Approval Date": "02/23/2021",
                "SS Awarded Date": None,
                "Primary Diagnosis Category": "Diseases of the musculoskeletal system & connectiv",
                "Taxable Pct": 0,
                "RTW Date": None,
                "Eligibility Outcome": "Approved",
                "SS Est Start Date": None,
                "SS Pri Status": "Unknown",
                "Plan Duration Date": None,
                "ThirdPartyReferralIndicator": None,
                "Primary Diagnosis Desc": "Other intervertebral disc displacement, lumbosacra",
                "Duration Date": None,
                "SocialSecurityPrimaryAwardType": None,
                "Gross Benefit Ind": 52.0,
                "Insured Hire Date": "04/30/2012",
                "Occ Category": None,
                "SubstanceAbuseDesc": None,
                "Insured Gender": "M",
                "Any Occ Category": "Own Occ",
                "Loss Date": "01/28/2021",
                "Voc Rehab Active Status": None,
                "Coverage Code": "STDATP",
                "SS Adjustment Ind": "N",
                "SS Eligible Ind": "Y",
                "Claim Status Code": "Open",
                "originalValues": "{'values': {'BenefitNumber': 'GDC-72418-01', 'ClaimStatusCategory': 'Open', 'ClaimCauseDescription': 'Accident', 'InsuredGender': 'Male', 'InsuredAnnualizedSalary': '85000-85250', 'GrossBenefitIndicator': 'Weekly', 'GrossBenefit': '750-1000', 'NetBenefit': '', 'EligibilityOutcome': 'Post Approval Rules', 'BenefitCaseType': 'STD', 'CaseSize': '2000-2025'}}",
                "Gross Benefit": 45500.0,
                "Insured Annualized Salary": 85125.0,
                "Net Benefit": None,
                "Policy Lives": 2012,
                "Servicing RSO": "Chicago",
                "Nurse Cert End Date": None,
            }
        },
    )
)
