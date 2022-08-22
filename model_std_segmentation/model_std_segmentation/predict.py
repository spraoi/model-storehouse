def predict(**kwargs):
    import functools

    import numpy as np
    import pandas as pd

    from model_std_segmentation.helpers import (
        _check_for_no_data,
        _check_na_counts_thresh,
        _compose,
        _date_diff_all,
        _fill_date_cols,
        _fill_unknown,
        _filter_bank_one,
        _filter_bank_two,
        _fix_nurse_date,
        _generate_payload,
        _payment_date_filter,
        _remap_features,
        _replace_state,
        _resolve_formatting,
        _to_category,
        categorical_cols,
        date_cols,
        download_model_from_s3,
        get_bucket_and_key_from_s3_uri,
        numeric_cols,
    )

    pd.options.mode.chained_assignment = None

    for artifact in kwargs.get("artifact"):
        if artifact.get("dataName") == "combined_artifacts":
            model_bucket, model_key = get_bucket_and_key_from_s3_uri(
                artifact.get("dataValue")
            )

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

    def _compose_data(df: pd.DataFrame):
        df.loc[:, scaler_cols] = robust_scaler_obj.transform(
            df.loc[:, scaler_cols].to_numpy()
        )

        df_enc = catboost_enc_obj.transform(df[categorical_cols])
        df.drop(columns=categorical_cols, inplace=True)
        df = pd.concat([df, df_enc], axis=1)

        return df

    def _output_transform_apply(df: pd.DataFrame, tier: int) -> pd.DataFrame:
        mapping = {
            1: "High potential RTW",
            2: "Complex RTW",
            3: "Behavioural health",
            4: "Extended disability",
            5: "Bad Data",
        }
        df["tier"] = tier
        df["tierHint"] = mapping[tier]
        if tier == 5:
            df["predictedProbability"] = np.NaN
            df["predictedSegment"] = np.NaN
        return df

    def _output_transform_logic(df: pd.DataFrame) -> pd.DataFrame:
        if df.loc[0, "predictedSegment"] == 0:
            df = _output_transform_apply(df, 1)
        elif df.loc[0, "predictedSegment"] == 1:
            x = df[df["pdCode"].isin(t4_list)]
            if not x.empty:
                df = _output_transform_apply(df, 4)
            else:
                df = _output_transform_apply(df, 2)
        return df

    artifacts = download_model_from_s3(model_bucket, model_key)

    (
        robust_scaler_obj,
        catboost_enc_obj,
        rf_model,
        template_obj,
        remap_obj,
        t4_list,
    ) = artifacts

    input_data = pd.DataFrame([kwargs.get("inputs").get("claim")])
    bkp_df = input_data.copy()
    claim_no = input_data["Claim Identifier"]
    pd_code = input_data["Primary Diagnosis Code"]
    pd_cat = input_data["Primary Diagnosis Category"]

    filter_data_fns = _compose(
        _payment_date_filter,
        _filter_bank_two,
        _check_na_counts_thresh,
        _filter_bank_one,
    )
    fix_data_fns = _compose(
        functools.partial(_replace_state, mapping=remap_obj),
        _to_category,
        _fill_unknown,
        _remap_features,
        _fix_nurse_date,
        _date_diff_all,
        _fill_date_cols,
        _resolve_formatting,
    )

    input_data = filter_data_fns(input_data)
    input_data = fix_data_fns(input_data)

    df = input_data[date_cols + numeric_cols + categorical_cols + ["bad_data"]]

    if _check_for_no_data(df) or df.loc[0, "bad_data"] == 1:
        return _generate_payload(_output_transform_apply(bkp_df, tier=5))

    df.drop(columns=["bad_data", "Loss Date"], inplace=True)
    df = _compose_data(df)
    prediction = rf_model.predict(df)
    class_confidence = float(rf_model.predict_proba(df)[0][prediction])
    df["predictedSegment"] = prediction
    df["predictedProbability"] = class_confidence
    df["Claim Identifier"] = claim_no
    df["pdCode"] = pd_code
    df["Primary Diagnosis Category"] = pd_cat

    payload = _generate_payload(_output_transform_logic(df))
    if df.loc[0, "Primary Diagnosis Category"] == "MENTAL & NERVOUS DISORDERS":
        payload = _generate_payload(_output_transform_apply(df, tier=3))
    return payload


# example input

# print(
#     predict(
#         model_name="model_std_segmentation",
#         artifact=[
#             {
#                 "dataId": "55dcc659-d0c5-42aa-b9bf-a0325a2997b9",
#                 "dataName": "combined_artifacts",
#                 "dataType": "artifact",
#                 "dataValue": "s3://spr-ml-artifacts/prod/segmentation/combined_artifacts_120k.sav",
#                 "dataValueType": "str",
#             }
#         ],
#         inputs={
#             "claim": {
#                 "Unnamed: 0": "21588",
#                 "Claim Identifier": "c",
#                 "Coverage Code": "GWI",
#                 "Received Date": 1485388800000,
#                 "Loss Date": 1455667200000,
#                 "Claim Status Category": "CLOSED",
#                 "Claim Status Code": "7R",
#                 "Claim Status Description": "CLOSED;MEDICAL CERTIFICATION NOT RECEIVED",
#                 "Claim State": "MA",
#                 "Closed Date": 1579564800000,
#                 "Claim Cause Desc": "OTHER ACCIDENT",
#                 "Certficate Num (Encoded)": None,
#                 "Insured Gender": "M",
#                 "Insured Age at Loss": 22.0,
#                 "Insured State": "NJ",
#                 "Insured Salary Ind": "H",
#                 "Insured Annualized Salary": 51476.0,
#                 "Insured Hire Date": 1376870400000,
#                 "Insured Zip": "71263",
#                 "Primary Diagnosis Code": "M67.471",
#                 "Primary Diagnosis Desc": "Open-angle glaucoma",
#                 "Primary Diagnosis Category": "MATERNITY",
#                 "Claim Diagnoses": "M54.6;Pain in thoracic spine;M43.06;Spondylolysis, lumbar region",
#                 "Mental Nervous Ind": None,
#                 "Gross Benefit": "528.0",
#                 "Net Benefit": "712.0",
#                 "Taxable Pct": "100",
#                 "RTW Date": None,
#                 "First Payment From Date": 1574899200000,
#                 "Last Payment To Date": 1581984000000,
#                 "Recovery Amt": None,
#                 "Approval Date": 1486512000000,
#                 "Duration Date": "01\\/15\\/2018",
#                 "Duration Months": "6.0",
#                 "Modified RTW Date": None,
#                 "Nurse Cert End Date": 1490659200000,
#                 "Eligibility Outcome": "APPROVED",
#                 "Elimination Days": "7",
#                 "Plan Benefit Max Amt": None,
#                 "Plan Benefit Pct": "0",
#                 "SIC Category": "Retail Trade",
#                 "SIC Code": "6531.0",
#                 "SIC Desc": "Pharmaceutical Preparations",
#                 "Occ Category": "Professionals",
#                 "DOT Desc": None,
#                 "DOT Code": None,
#                 "DOT Exertion Level (Primary)": None,
#                 "Client ID": "PIT20110616115718",
#                 "Client ZI˝P": "55416",
#                 "Policy Number": "VPS326440",
#                 "Policy Line of Business": "VPS",
#                 "Policy Effective Date": 1167609600000,
#                 "Policy Termination Date": 1546300800000,
#                 "Annualized Premium": "587803.0",
#                 "Policy Lives": 4359,
#                 "Case Size": "0-100",
#                 "IEB Ind": "NOT IEB",
#                 "Servicing RSO": "Atlanta",
#                 "Examiner Name": "ORTIZ, NYREE",
#                 "User Examiner": "PAONESSA, NIKKI",
#                 "Any Occ Category": None,
#                 "Any Occ Start Date": "3\\/13\\/2017",
#                 "Any Occ Decision Due Date": None,
#                 "Any Occ Decision Made Date": None,
#                 "SS Accept for Rep": None,
#                 "SS Eligible Ind": "N",
#                 "SS Pursue Ind": None,
#                 "SS Pri Status": None,
#                 "SS Pri Award Amt": None,
#                 "SS Pri Award Type": None,
#                 "SS Pri Award Eff Date": None,
#                 "SS Adjustment Ind": "Y",
#                 "SS Reject Code": None,
#                 "SS Pri Est Offset Amt": None,
#                 "SS Est Monthly Benefit Amt": None,
#                 "SS Est Start Date": None,
#                 "SS Awarded Date": None,
#                 "SS Remarks": None,
#                 "Voc Rehab Status": None,
#                 "Voc Rehab Service Requested": None,
#                 "Pre-Ex Outcome": "NO",
#                 "Pre-Ex Investigation Ind": None,
#                 "TSA Ind": "N",
#                 "Voc Rehab TSA Date": None,
#                 "Tier": None,
#                 "Tier Description": None,
#                 "Potential Resolution Date": None,
#                 "WFAM Code": None,
#                 "STD to LTD Bridge Ind": None,
#                 "Associated LTD Policy Ind": None,
#                 "SS Dep Off Set Allowed": None,
#                 "SS Primary Hardship Indicator": "N",
#                 "Unnamed: 91": None,
#                 "resolve": "genuine",
#                 "predictionValue": "0.0",
#                 "bad_data": 0,
#             },
#             "__row_number": 1,
#             "__root_row_number": 1,
#         },
#     )
# )
