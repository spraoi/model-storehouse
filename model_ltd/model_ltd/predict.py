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
            model_bucket, model_key = get_bucket_and_key_from_s3_uri(
                artifact.get("dataValue")
            )
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
    if test_X.loc[[na_ind == "N" for na_ind in na_inds], :].shape[0]:
        predict_prob = loaded_model.predict_proba(
            test_X.loc[[na_ind == "N" for na_ind in na_inds], :].apply(np.nan_to_num)
        )[:, 1]
    else:
        predict_prob = None
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
        {
            "inputDataSource": f"{predicted_claim.get('claimNumber')}:0",
            "entityId": predicted_claim.get("claimNumber"),
            "predictedResult": predicted_claim,
        }
    ]

# DO NOT UNCOMMENT THIS CODE EXCEPT FOR LOCAL TESTING!
# print(
#     predict(
#         model_name="RF_LTD_Fraud_model",
#         artifact=[
#             {
#                 "dataId": "5ad3e9c0-b248-4397-89a9-f44f3d3b7454",
#                 "dataName": "schema_artifact",
#                 "dataType": "artifact",
#                 "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/train_schema_2021-07-09.pbtxt",
#                 "dataValueType": "str",
#             },
#             {
#                 "dataId": "06335936-85fd-4c70-8d22-6cb9686be859",
#                 "dataName": "statistics_artifact",
#                 "dataType": "artifact",
#                 "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/train_stats_2021-07-09.pbtxt",
#                 "dataValueType": "str",
#             },
#             {
#                 "dataId": "3e1bbfda-3320-45b4-8af3-ad4a7fbaeb7a",
#                 "dataName": "feat_imp_artifact",
#                 "dataType": "artifact",
#                 "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/feature_importances_2021-07-09.csv",
#                 "dataValueType": "str",
#             },
#             {
#                 "dataId": "7d5b2be0-435e-437b-bfd2-f86a7b0c836b",
#                 "dataName": "template_artifact",
#                 "dataType": "artifact",
#                 "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/template_data_2021-07-09.csv",
#                 "dataValueType": "str",
#             },
#             {
#                 "dataId": "68b31557-8a93-44ab-97de-a936b398f541",
#                 "dataName": "model_artifact",
#                 "dataType": "artifact",
#                 "dataValue": "s3://spr-ml-artifacts/prod/RF_LTD_Fraud_Model/artifacts/model_2021-07-09.joblib",
#                 "dataValueType": "str",
#             },
#         ],
#         inputs={
#             "claim": {
#                 "Claim Number": "GDC-1728",
#                 "Claim Identifier": "GDC-1728-02",
#                 "Policy Line of Business": "LTD",
#                 "Coverage Code": "LTD",
#                 "Received Date": "09/10/2020",
#                 "Loss Date": "09/01/2020",
#                 "Claim Status Category": "ACTIVE",
#                 "Claim Status Code": "Open",
#                 "Claim Status Description": "Benefit Case Under Review",
#                 "Claim State": "IL",
#                 "Closed Date": None,
#                 "Claim Cause Desc": "OTHER SICKNESS",
#                 "Insured Gender": "M",
#                 "Insured Age at Loss": "20",
#                 "Insured State": "IL",
#                 "Insured Salary Ind": "WEEKLY",
#                 "Insured Annualized Salary": 70125,
#                 "Insured Hire Date": "05/12/2015",
#                 "Insured Zip": "60441",
#                 "Primary Diagnosis Code": None,
#                 "Primary Diagnosis Desc": None,
#                 "Primary Diagnosis Category": None,
#                 "Secondary Diagnosis Code": None,
#                 "Secondary Diagnosis Desc": None,
#                 "Secondary Diagnosis Category": None,
#                 "Secondary Diagnosis Category Desc": None,
#                 "Mental Nervous Ind": None,
#                 "MentalNervousDesc": None,
#                 "SubstanceAbuseDesc": None,
#                 "Gross Benefit Ind": 12,
#                 "Gross Benefit": 34500,
#                 "Net Benefit": 0,
#                 "Taxable Pct": "0",
#                 "RTW Date": None,
#                 "First Payment From Date": "09/08/2020",
#                 "Last Payment To Date": "09/08/2020",
#                 "Recovery Amt": " 0000000000000000000000.000000",
#                 "Approval Date": "07/20/2021",
#                 "Duration Date": None,
#                 "Duration Months": "19",
#                 "Modified RTW Date": None,
#                 "Expected Term Date": None,
#                 "Clinical End Date": None,
#                 "Eligibility Outcome": "Approved",
#                 "Elimination Period": None,
#                 "Elimination Ind": None,
#                 "Elimination Days": "0",
#                 "Plan Duration Date": None,
#                 "Plan Benefit Pct": "0.5",
#                 "SIC Category": None,
#                 "SIC Code": "8412",
#                 "SIC Desc": "MUSEUMS AND ART GALLERIES",
#                 "Occ Category": "Computer Programmer,Information Technologies (IT)",
#                 "Occ Code": "05",
#                 "DOT Code": None,
#                 "DOT Desc": None,
#                 "DOT Exertion Level (Primary)": "Light",
#                 "Policy Effective Date": "01/01/2017",
#                 "Policy Termination Date": None,
#                 "Policy Lives": 62,
#                 "Any Occ Category": "Unknown",
#                 "Any Occ Ind": None,
#                 "Any Occ period": None,
#                 "Any Occ Start Date": "03/01/2022",
#                 "Any Occ Decision Due Date": None,
#                 "Any Occ Decision Made Date": None,
#                 "SS Eligible Ind": "Y",
#                 "SS Pursue Ind": None,
#                 "SS Pri Status": "Unknown",
#                 "SS Pri Award Amt": None,
#                 "SS Pri Award Eff Date": None,
#                 "SS Adjustment Ind": "N",
#                 "SS Reject Code": None,
#                 "SS Est Start Date": None,
#                 "SS Awarded Date": None,
#                 "Voc Rehab Status": None,
#                 "Voc Rehab Service Requested": None,
#                 "Voc Rehab Active Status": None,
#                 "Voc Rehab Outcome": None,
#                 "Pre-Ex Outcome": "Y",
#                 "TSA Ind": "N",
#                 "Voc Rehab TSA Date": None,
#                 "originalValues": "{'BenefitNumber': 'GDC-1728-02', 'ClaimStatusCategory': 'Open', 'ClaimCauseDescription': 'Sickness', 'InsuredGender': 'Male', 'InsuredAnnualizedSalary': '70000-70250', 'GrossBenefitIndicator': 'Monthly', 'GrossBenefit': '2750-3000', 'NetBenefit': '-NA-', 'EligibilityOutcome': 'Post Approval Rules', 'BenefitCaseType': 'LTD', 'CaseSize': '50-75'}",
#                 "Servicing RSO": "Chicago",
#                 "Nurse Cert End Date": None,
#             }
#         },
#     )
# )
