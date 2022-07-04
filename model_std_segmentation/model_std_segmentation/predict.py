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
            if df.loc[0, "pdCode"].isin(t4_list):
                df = _output_transform_apply(df, 4)
            else:
                df = _output_transform_apply(df, 2)
        return df

    artifacts = download_model_from_s3(model_bucket, model_key)
    # with open("./data/combined_artifacts_120k.sav", "rb") as f:
    #     artifacts = joblib.load(f)
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

    payload = _generate_payload(_output_transform_logic(df))
    if df.loc[0, "pdCode"] == "Mental and Nervous Disorder":
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
#                 "Mental Nervous Ind": None,
#                 "Recovery Amt": 0.0,
#                 "Modified RTW Date": None,
#                 "Any Occ period": None,
#                 "__root_row_number": 366,
#                 "Claim Number": "GDC-72418",
#                 "Policy Effective Date": "01/01/2017",
#                 "DOT Exertion Level (Primary)": "Unknown",
#                 "Last Payment To Date": "01/28/2021",
#                 "DOT Desc": None,
#                 "Elimination Period": None,
#                 "DOT Code": None,
#                 "Voc Rehab Outcome": None,
#                 "Policy Line of Business": "STD",
#                 "Expected Term Date": None,
#                 "Clinical End Date": None,
#                 "Voc Rehab Service Requested": None,
#                 "Policy Termination Date": None,
#                 "Any Occ Start Date": None,
#                 "Duration Months": 1,
#                 "MentalNervousDesc": None,
#                 "Closed Date": None,
#                 "Insured State": "CA",
#                 "SS Pri Award Amt": None,
#                 "Any Occ Decision Due Date": None,
#                 "Elimination Days": 0,
#                 "SS Pursue Ind": None,
#                 "Any Occ Ind": None,
#                 "Elimination Ind": None,
#                 "__row_number": 366,
#                 "Plan Benefit Pct": 0.6,
#                 "Claim Status Description": "Benefit Case Under Review",
#                 "Secondary Diagnosis Code": "M54.17",
#                 "Secondary Diagnosis Category": None,
#                 "Claim Identifier": "GDC-72418-01",
#                 "SS Pri Award Eff Date": None,
#                 "Pre-Ex Outcome": "Y",
#                 "Claim Status Category": "ACTIVE",
#                 "Primary Diagnosis Code": "M51.27",
#                 "Voc Rehab Status": None,
#                 "Claim Cause Desc": "OTHER ACCIDENT",
#                 "Insured Salary Ind": "BI-WEEKLY",
#                 "Insured Zip": "93447",
#                 "SIC Code": 7349,
#                 "First Payment From Date": "01/28/2021",
#                 "SS Reject Code": None,
#                 "Any Occ Decision Made Date": None,
#                 "SIC Category": None,
#                 "Insured Age at Loss": 53,
#                 "Received Date": "02/12/2021",
#                 "Secondary Diagnosis Desc": "Radiculopathy, lumbosacral region",
#                 "Voc Rehab TSA Date": None,
#                 "TSA Ind": "N",
#                 "Secondary Diagnosis Category Desc": None,
#                 "SIC Desc": "BUILDING MAINTENANCE SERVICES, NEC",
#                 "Claim State": "CA",
#                 "ThirdPartyReferralDescription": None,
#                 "Occ Code": None,
#                 "Approval Date": "02/23/2021",
#                 "SS Awarded Date": None,
#                 "Primary Diagnosis Category": "Diseases of the musculoskeletal system & connectiv",
#                 "Taxable Pct": 0,
#                 "RTW Date": None,
#                 "Eligibility Outcome": "Approved",
#                 "SS Est Start Date": None,
#                 "SS Pri Status": "Unknown",
#                 "Plan Duration Date": None,
#                 "ThirdPartyReferralIndicator": None,
#                 "Primary Diagnosis Desc": "Other intervertebral disc displacement, lumbosacra",
#                 "Duration Date": None,
#                 "SocialSecurityPrimaryAwardType": None,
#                 "Gross Benefit Ind": 52.0,
#                 "Insured Hire Date": "04/30/2012",
#                 "Occ Category": None,
#                 "SubstanceAbuseDesc": None,
#                 "Insured Gender": "M",
#                 "Any Occ Category": "Own Occ",
#                 "Loss Date": "01/28/2021",
#                 "Voc Rehab Active Status": None,
#                 "Coverage Code": "STDATP",
#                 "SS Adjustment Ind": "N",
#                 "SS Eligible Ind": "Y",
#                 "Claim Status Code": "Open",
#                 "originalValues": "{'values': {'BenefitNumber': 'GDC-72418-01', 'ClaimStatusCategory': 'Open', 'ClaimCauseDescription': 'Accident', 'InsuredGender': 'Male', 'InsuredAnnualizedSalary': '85000-85250', 'GrossBenefitIndicator': 'Weekly', 'GrossBenefit': '750-1000', 'NetBenefit': '', 'EligibilityOutcome': 'Post Approval Rules', 'BenefitCaseType': 'STD', 'CaseSize': '2000-2025'}}",
#                 "Gross Benefit": 45500.0,
#                 "Insured Annualized Salary": 85125.0,
#                 "Net Benefit": None,
#                 "Policy Lives": 2012,
#                 "Servicing RSO": "Chicago",
#                 "Nurse Cert End Date": None,
#             }
#         },
#     )
# )
