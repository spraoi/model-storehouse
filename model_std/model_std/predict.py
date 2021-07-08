def predict(**kwargs):

    import pandas as pd
    import numpy as np
    from model_std.helpers import (
        magnificent_map,
        resolve_formatting,
        add_policy_tenure_to_df,
        add_days_rep_to_df,
        add_emp_tenure_to_df,
        add_prognosis_days_to_df,
        to_category,
        test_train_match,
        get_na_rows,
        posterior_correction,
        download_obj_from_s3,
        get_bucket_and_key_from_s3_uri
    )
    import json


    p1_orig = 0.00795
    p_1_train = 0.1792  # 1: {0: 24995, 1: 5458}

    columns = [
        "Claim Identifier",
        "Received Date",
        "Loss Date",
        "Insured Gender",
        "Insured Age at Loss",
        "Insured Salary Ind",
        "Insured Hire Date",
        "Insured Zip",
        "Primary Diagnosis Category",
        "Duration Date",
        "Duration Months",
        "SIC Code",
        "Coverage Code",
        "Policy Termination Date",
        "Policy Effective Date",
        "Insured Annualized Salary",
        "Policy Lives",
    ]

    dates = [
        "Received Date",
        "Loss Date",
        "Insured Hire Date",
        "Duration Date",
        "Policy Effective Date",
        "Policy Termination Date",
    ]
    numeric = ["Insured Age at Loss", "Insured Annualized Salary", "Policy Lives"]
    categorical = [
        "Insured Gender",
        "Insured Salary Ind",
        "Primary Diagnosis Category",
        "SIC_risk_ind",
        "Coverage Code",
    ]

    artifacts = kwargs.get("artifacts")

    model_bucket, model_key = get_bucket_and_key_from_s3_uri(artifacts.get("model"))
    loaded_model = download_obj_from_s3(model_bucket, model_key)

    loaded_template = pd.read_csv(artifacts.get('template'))

    magnificent_data = pd.DataFrame([kwargs.get("inputs").get("claim")])
    test_data = magnificent_data[columns].copy()
    test_data = magnificent_map(
        test_data,
        remap_columns=[
            "Insured Salary Ind",
            "SIC Code",
            "Coverage Code",
            "Primary Diagnosis Category",
        ],
    )
    test_data = resolve_formatting(test_data, dates, numeric)
    test_data = add_policy_tenure_to_df(test_data)
    test_data = add_days_rep_to_df(test_data).copy()
    test_data.loc[test_data["days_to_report"] < 0, "days_to_report"] = 0
    test_data = add_emp_tenure_to_df(test_data)
    test_data = add_prognosis_days_to_df(test_data)
    test_data = test_data.select_dtypes(exclude=["datetime64"])
    test_data.drop(["Insured Zip", "Duration Months"], axis=1, inplace=True)
    test_data = to_category(test_data, categorical)
    test_data = get_na_rows(test_data)




    na_inds = list(test_data.pop("NA_row"))
    test_data = pd.get_dummies(
        test_data, columns=test_data.select_dtypes(include="category").columns
    )
    test_X = test_train_match(loaded_template, test_data)
    predict_prob = loaded_model.predict_proba(
        test_X.loc[[na_ind == "N" for na_ind in na_inds], :]
    )[:, 1]
    test_final = magnificent_data.loc[
        :, ["Claim Identifier", "Claim Status Category"]
    ].copy()
    test_final.loc[[na_ind == "N" for na_ind in na_inds], "probability"] = predict_prob
    test_final.loc[[na_ind == "Y" for na_ind in na_inds], "probability"] = np.nan
    test_final.loc[:, "p_corrected"] = posterior_correction(
        p1_orig, p_1_train, test_final.loc[:, "probability"]
    )
    test_final.loc[:, "p_labels_corrected"] = (
        test_final.loc[:, "p_corrected"] >= 0.5
    ).astype(int)
    test_final.loc[[na_ind == "Y" for na_ind in na_inds], "p_labels_corrected"] = np.nan

    test_final.loc[:, "p_labels_corrected"] = test_final.loc[
        :, "p_labels_corrected"
    ].astype("Int64")

    payload_data = test_final.loc[
        :, ["Claim Identifier", "p_corrected", "p_labels_corrected"]
    ].copy()  # ,'Claim Status Category'
    payload_data.columns = [
        "claimNumber",
        "predictedProbability",
        "predictedValue",
    ]  # ,'claimStatusCategory'
    prediction_json = json.loads(payload_data.to_json(orient="records"))
    predicted_claim = prediction_json[0] if prediction_json else None
    return [{"inputDataSource":f"{predicted_claim['claimNumber']}:0","entityId":predicted_claim["claimNumber"],"predictedResult":predicted_claim}]


#example input

# print(predict(model_name="model_std",artifacts={"model":"s3://spr-ml-artifacts/prod/MLMR_STD_Fraud_Model/artifacts/model_2021-06-30.joblib","template":"s3://spr-ml-artifacts/prod/MLMR_STD_Fraud_Model/artifacts/template_data_2021-06-30.csv"},inputs={"claim":
#     {
#         "Mental Nervous Ind": None,
#         "Recovery Amt": 0.0,
#         "Modified RTW Date": None,
#         "Any Occ period": None,
#         "__root_row_number": 366,
#         "Claim Number": "GDC-72418",
#         "Policy Effective Date": "01/01/2017",
#         "DOT Exertion Level (Primary)": "Unknown",
#         "Last Payment To Date": "01/28/2021",
#         "DOT Desc": None,
#         "Elimination Period": None,
#         "DOT Code": None,
#         "Voc Rehab Outcome": None,
#         "Policy Line of Business": "STD",
#         "Expected Term Date": None,
#         "Clinical End Date": None,
#         "Voc Rehab Service Requested": None,
#         "Policy Termination Date": None,
#         "Any Occ Start Date": None,
#         "Duration Months": 1,
#         "MentalNervousDesc": None,
#         "Closed Date": None,
#         "Insured State": "CA",
#         "SS Pri Award Amt": None,
#         "Any Occ Decision Due Date": None,
#         "Elimination Days": 0,
#         "SS Pursue Ind": None,
#         "Any Occ Ind": None,
#         "Elimination Ind": None,
#         "__row_number": 366,
#         "Plan Benefit Pct": 0.6,
#         "Claim Status Description": "Benefit Case Under Review",
#         "Secondary Diagnosis Code": "M54.17",
#         "Secondary Diagnosis Category": None,
#         "Claim Identifier": "GDC-72418-01",
#         "SS Pri Award Eff Date": None,
#         "Pre-Ex Outcome": "Y",
#         "Claim Status Category": "ACTIVE",
#         "Primary Diagnosis Code": "M51.27",
#         "Voc Rehab Status": None,
#         "Claim Cause Desc": "OTHER ACCIDENT",
#         "Insured Salary Ind": "BI-WEEKLY",
#         "Insured Zip": "93447",
#         "SIC Code": 7349,
#         "First Payment From Date": "01\/28\/2021",
#         "SS Reject Code": None,
#         "Any Occ Decision Made Date": None,
#         "SIC Category": None,
#         "Insured Age at Loss": 53,
#         "Received Date": "02/12/2021",
#         "Secondary Diagnosis Desc": "Radiculopathy, lumbosacral region",
#         "Voc Rehab TSA Date": None,
#         "TSA Ind": "N",
#         "Secondary Diagnosis Category Desc": None,
#         "SIC Desc": "BUILDING MAINTENANCE SERVICES, NEC",
#         "Claim State": "CA",
#         "ThirdPartyReferralDescription": None,
#         "Occ Code": None,
#         "Approval Date": "02/23/2021",
#         "SS Awarded Date": None,
#         "Primary Diagnosis Category": "Diseases of the musculoskeletal system & connectiv",
#         "Taxable Pct": 0,
#         "RTW Date": None,
#         "Eligibility Outcome": "Approved",
#         "SS Est Start Date": None,
#         "SS Pri Status": "Unknown",
#         "Plan Duration Date": None,
#         "ThirdPartyReferralIndicator": None,
#         "Primary Diagnosis Desc": "Other intervertebral disc displacement, lumbosacra",
#         "Duration Date": None,
#         "SocialSecurityPrimaryAwardType": None,
#         "Gross Benefit Ind": 52.0,
#         "Insured Hire Date": "04/30/2012",
#         "Occ Category": None,
#         "SubstanceAbuseDesc": None,
#         "Insured Gender": "M",
#         "Any Occ Category": "Own Occ",
#         "Loss Date": "01/28/2021",
#         "Voc Rehab Active Status": None,
#         "Coverage Code": "STDATP",
#         "SS Adjustment Ind": "N",
#         "SS Eligible Ind": "Y",
#         "Claim Status Code": "Open",
#         "originalValues": "{'values': {'BenefitNumber': 'GDC-72418-01', 'ClaimStatusCategory': 'Open', 'ClaimCauseDescription': 'Accident', 'InsuredGender': 'Male', 'InsuredAnnualizedSalary': '85000-85250', 'GrossBenefitIndicator': 'Weekly', 'GrossBenefit': '750-1000', 'NetBenefit': '', 'EligibilityOutcome': 'Post Approval Rules', 'BenefitCaseType': 'STD', 'CaseSize': '2000-2025'}}",
#         "Gross Benefit": 45500.0,
#         "Insured Annualized Salary": 85125.0,
#         "Net Benefit": None,
#         "Policy Lives": 2012,
#         "Servicing RSO": "Chicago",
#         "Nurse Cert End Date": None
#     }}))
