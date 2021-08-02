def predict(**kwargs):

    import pandas as pd
    import numpy as np
    from model_ltd.helpers import (
        magnificent_map,
        resolve_formatting,
        get_date_diff,
        add_emp_tenure_to_df,
        add_payment_lag_to_df,
        add_prognosis_days_to_df,
        to_category,
        test_train_match,
        get_na_rows,
        download_obj_from_s3,
        get_bucket_and_key_from_s3_uri,
    )
    import json

    COLUMNS = [
        "Claim Identifier",
        "Claim Cause Desc",
        "Claim State",
        "Claim Status Category",
        "Claim Status Code",
        "Coverage Code",
        "DOT Exertion Level (Primary)",
        "Eligibility Outcome",
        "Insured Gender",
        "Insured State",
        "Mental Nervous Ind",
        "Pre-Ex Outcome",
        "Occ Category",
        "Primary Diagnosis Category",
        "Primary Diagnosis Code",
        "Primary Diagnosis Desc",
        "SIC Category",
        "SS Adjustment Ind",
        "SS Eligible Ind",
        "SS Pursue Ind",
        "SS Reject Code",
        "Voc Rehab Active Status",
        "Voc Rehab Outcome",
        "First Payment From Date",
        "Last Payment To Date",
        "Duration Date",
        "Loss Date",
        "Insured Hire Date",
        "Policy Effective Date",
        "Policy Termination Date",
        "Received Date",
        "Insured Age at Loss",
        "Insured Annualized Salary",
        "Insured Salary Ind",
        "Policy Lives",
        "Duration Months",
    ]

    DATES = [
        "First Payment From Date",
        "Last Payment To Date",
        "Duration Date",
        "Loss Date",
        "Insured Hire Date",
        "Policy Effective Date",
        "Policy Termination Date",
        "Received Date",
    ]

    NUMERIC = [
        "Insured Age at Loss",
        "Insured Annualized Salary",
        "Policy Lives",
        "Duration Months",
    ]
    CATEGORICAL = [
        "Insured Salary Ind",
        "Claim Cause Desc",
        "Claim State",
        "Coverage Code",
        "DOT Exertion Level (Primary)",
        "Eligibility Outcome",
        "Insured Gender",
        "Insured State",
        "Mental Nervous Ind",
        "Pre-Ex Outcome",
        "Occ Category",
        "Primary Diagnosis Category",
        "Primary Diagnosis Code",
        "Primary Diagnosis Desc",
        "SIC Category",
        "SS Adjustment Ind",
        "SS Eligible Ind",
        "SS Pursue Ind",
        "SS Reject Code",
        "Voc Rehab Active Status",
        "Voc Rehab Outcome",
    ]

    for artifact in kwargs.get("artifact"):
        if artifact.get("dataName") == "model_artifact":
            model_bucket, model_key = get_bucket_and_key_from_s3_uri(artifact.get("dataValue"))
        elif artifact.get("dataName") == "template_artifact":
            loaded_template = pd.read_csv(artifact.get("dataValue"))

    loaded_model = download_obj_from_s3(model_bucket, model_key)


    magnificent_data = pd.DataFrame([kwargs.get("inputs").get("claim")])
    test_data = magnificent_data[COLUMNS].copy()

    test_data = resolve_formatting(test_data, DATES, NUMERIC)
    test_data["days_to_report"] = get_date_diff(
        test_data["Loss Date"], test_data["Received Date"], interval="D"
    )
    test_data = add_emp_tenure_to_df(test_data)

    test_data = add_payment_lag_to_df(test_data)
    test_data = add_prognosis_days_to_df(test_data)

    test_data.loc[test_data["days_to_report"] < 0, "days_to_report"] = np.nan
    test_data.loc[
        test_data["last_first_payment_lag"] < 0, "last_first_payment_lag"
    ] = np.nan
    test_data.drop(["Claim Status Category"], axis=1, inplace=True)

    test_data = magnificent_map(
        test_data,
        remap_columns=[
            "Insured Salary Ind",
            "Pre-Ex Outcome",
            "SS Pursue Ind",
        ],
    )

    test_data = test_data.select_dtypes(exclude=["datetime64"])
    test_data = to_category(test_data, CATEGORICAL)
    test_data = get_na_rows(test_data)

    na_inds = list(test_data.pop("NA_row"))
    test_data = pd.get_dummies(
        test_data, columns=test_data.select_dtypes(include="category").columns
    )

    test_X = test_train_match(loaded_template, test_data)
    predict_prob = loaded_model.predict_proba(
        test_X.loc[[na_ind == "N" for na_ind in na_inds], :].apply(np.nan_to_num)
    )[:, 1]

    test_final = magnificent_data.loc[
        :, ["Claim Identifier", "Claim Status Category"]
    ].copy()
    test_final.loc[[na_ind == "N" for na_ind in na_inds], "probability"] = predict_prob
    test_final.loc[[na_ind == "Y" for na_ind in na_inds], "probability"] = np.nan
    # disabling posterior correction for now
    # test_final.loc[:, "p_corrected"] = posterior_correction(
    #     P1_ORIG, P1_TRAIN, test_final.loc[:, "probability"]
    # )
    # test_final.loc[:, "p_labels_corrected"] = (
    #     test_final.loc[:, "p_corrected"] >= 0.5
    # ).astype(int)
    # test_final.loc[[na_ind == "Y" for na_ind in na_inds], "p_labels_corrected"] = np.nan

    # test_final.loc[:, "p_labels_corrected"] = test_final.loc[
    #     :, "p_labels_corrected"
    # ].astype("Int64")
    test_final.loc[:, "p_labels"] = (test_final.loc[:, "probability"] >= 0.5).astype(
        "Int64"
    )
    test_final.loc[[na_ind == "Y" for na_ind in na_inds], "p_labels"] = np.nan

    payload_data = test_final.loc[
        :, ["Claim Identifier", "probability", "p_labels"]
    ].copy()
    payload_data.columns = [
        "claimNumber",
        "predictedProbability",
        "predictedValue",
    ]
    prediction_json = json.loads(payload_data.to_json(orient="records"))
    predicted_claim = prediction_json[0] if prediction_json else None
    return [
        {"inputDataSource": f"{predicted_claim.get('claimNumber')}:0", "entityId": predicted_claim.get("claimNumber"),
         "predictedResult": predicted_claim}]



# print(
#     predict(
#         model_name="RF_LTD_Fraud_model",
#         artifact=[
#         {
#             "dataId": "5ad3e9c0-b248-4397-89a9-f44f3d3b7454",
#             "dataName": "schema_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/train_schema_2021-07-09.pbtxt",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "06335936-85fd-4c70-8d22-6cb9686be859",
#             "dataName": "statistics_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/train_stats_2021-07-09.pbtxt",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "3e1bbfda-3320-45b4-8af3-ad4a7fbaeb7a",
#             "dataName": "feat_imp_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/feature_importances_2021-07-09.csv",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "7d5b2be0-435e-437b-bfd2-f86a7b0c836b",
#             "dataName": "template_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/template_data_2021-07-09.csv",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "68b31557-8a93-44ab-97de-a936b398f541",
#             "dataName": "model_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/model_2021-07-09.joblib",
#             "dataValueType": "str"
#         }
#     ],
#         inputs={
#             "claim":
#                 {
#                     "Mental Nervous Ind": "",
#                     "Recovery Amt": " 0000000000000000000000.000000",
#                     "Modified RTW Date": None,
#                     "Any Occ period": "",
#                     "Claim Number": "GDC-46016",
#                     "Policy Effective Date": "01/01/2017",
#                     "DOT Exertion Level (Primary)": "Unknown",
#                     "Last Payment To Date": "08/29/2020",
#                     "DOT Desc": "",
#                     "Elimination Period": "",
#                     "DOT Code": "",
#                     "Voc Rehab Outcome": "",
#                     "Policy Line of Business": "LTD",
#                     "Expected Term Date": None,
#                     "Clinical End Date": None,
#                     "Voc Rehab Service Requested": "",
#                     "Policy Termination Date": None,
#                     "Any Occ Start Date": "03/02/2023",
#                     "Duration Months": "12",
#                     "MentalNervousDesc": "",
#                     "Closed Date": None,
#                     "Insured State": "IL",
#                     "SS Pri Award Amt": "-(2000-2250)",
#                     "Any Occ Decision Due Date": None,
#                     "Elimination Days": "0",
#                     "SS Pursue Ind": "",
#                     "Any Occ Ind": "",
#                     "Elimination Ind": "",
#                     "Plan Benefit Pct": "0.6667",
#                     "Claim Status Description": "Benefit Case Under Review",
#                     "Secondary Diagnosis Code": "",
#                     "Secondary Diagnosis Category": "",
#                     "Claim Identifier": "GDC-46016",
#                     "SS Pri Award Eff Date": "09/01/2020",
#                     "Pre-Ex Outcome": "Y",
#                     "Claim Status Category": "ACTIVE",
#                     "Primary Diagnosis Code": "J84.9",
#                     "Voc Rehab Status": "",
#                     "Claim Cause Desc": "OTHER SICKNESS",
#                     "Insured Salary Ind": "WEEKLY",
#                     "Insured Zip": "60438",
#                     "SIC Code": "8069",
#                     "First Payment From Date": "08/29/2020",
#                     "SS Reject Code": "",
#                     "Any Occ Decision Made Date": "08/13/2020",
#                     "SIC Category": "",
#                     "Insured Age at Loss": "63",
#                     "Received Date": "03/12/2020",
#                     "Secondary Diagnosis Desc": "",
#                     "Voc Rehab TSA Date": None,
#                     "TSA Ind": "N",
#                     "Secondary Diagnosis Category Desc": "",
#                     "SIC Desc": "SPECIALTY HOSPITAL EXC., PSYCHIATRIC",
#                     "Claim State": "IL",
#                     "ThirdPartyReferralDescription": "",
#                     "Occ Code": "",
#                     "Approval Date": "08/13/2020",
#                     "SS Awarded Date": None,
#                     "Primary Diagnosis Category": "Diseases of the respiratory system",
#                     "Taxable Pct": "0",
#                     "RTW Date": None,
#                     "Eligibility Outcome": "Approved",
#                     "SS Est Start Date": None,
#                     "SS Pri Status": "Approved",
#                     "Plan Duration Date": None,
#                     "ThirdPartyReferralIndicator": "",
#                     "Primary Diagnosis Desc": "Interstitial pulmonary disease, unspecified",
#                     "Duration Date": "08/28/2021",
#                     "SocialSecurityPrimaryAwardType": "Primary Social Security Retirement with Freeze",
#                     "Gross Benefit Ind": 12,
#                     "Insured Hire Date": "04/30/1985",
#                     "Occ Category": "",
#                     "SubstanceAbuseDesc": "",
#                     "Insured Gender": "M",
#                     "Any Occ Category": "Own Occ",
#                     "Loss Date": "03/02/2020",
#                     "Voc Rehab Active Status": "",
#                     "Coverage Code": "LTD",
#                     "SS Adjustment Ind": "Y",
#                     "SS Eligible Ind": "Y",
#                     "Claim Status Code": "Open",
#                     "Gross Benefit": 25500.0,
#                     "Insured Annualized Salary": 38875.0,
#                     "Net Benefit": 1500.0,
#                     "Policy Lives": 412.0,
#                     "Servicing RSO": "Chicago",
#                     "Nurse Cert End Date": None,
#                 }
#         },
#     )
# )
