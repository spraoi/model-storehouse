def predict(**kwargs):
    import pandas as pd
    import numpy as np
    import os

    from model_settlement.helpers import (
        pre_process_data,
        add_policy_tenure_to_df,
        add_days_rep_to_df,
        add_emp_tenure_to_df,
        add_prognosis_days_to_df,
        add_first_payment_recd_date_days_to_df,
        scale_features,
        get_na_rows,
        map_categories,
        map_diag_entities,
        test_train_match,
        get_bucket_and_key_from_s3_uri,
        download_obj_from_s3,
        posterior_correction,
    )

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    P1_ORIG = 0.1
    P1_TRAIN = 0.5

    artifacts = kwargs.get("artifacts")

    model_bucket, dl_model_key = get_bucket_and_key_from_s3_uri(
        artifacts.get("dl_model")
    )

    xgb_model_key = get_bucket_and_key_from_s3_uri(artifacts.get("xgb_model"))[1]
    scaler_key = get_bucket_and_key_from_s3_uri(artifacts.get("scaler"))[1]
    template_key = get_bucket_and_key_from_s3_uri(artifacts.get("template"))[1]

    loaded_dl_model = download_obj_from_s3(model_bucket, dl_model_key, "dl_model")

    loaded_scaler = download_obj_from_s3(model_bucket, scaler_key, "scaler")

    loaded_xgb = download_obj_from_s3(model_bucket, xgb_model_key, "xgb_model")

    loaded_template = pd.read_csv(f"s3://{model_bucket}/{template_key}")

    data = pd.DataFrame([kwargs.get("inputs").get("claim")])

    data = data.replace(
        r"^\s*$", np.nan, regex=True
    )  # for replacing empty strings with nans ""

    # data = pd.read_csv("magni_sett_data_mar_2021.csv")
    label_data = pd.read_csv("settlement_labels.csv")
    data_sub = data.loc[(~data["Claim Identifier"].isin(label_data["claimNumber"])), :]

    settlement = pre_process_data(data_sub)
    settlement = add_policy_tenure_to_df(settlement)
    settlement = add_days_rep_to_df(settlement).copy()
    settlement = add_emp_tenure_to_df(settlement)
    settlement = add_prognosis_days_to_df(settlement)
    settlement = add_first_payment_recd_date_days_to_df(settlement)
    settlement = settlement.select_dtypes(exclude=["datetime64"])
    settlement = scale_features(settlement, loaded_scaler)
    settlement.drop(["Duration Months"], axis=1, inplace=True)
    # serving_df = settlement.copy() for skew-analysis (TO DO)
    settlement = get_na_rows(settlement)

    sett_mapped = map_categories(settlement)

    na_inds = list(settlement.pop("NA_row"))
    TO_DUMMIES = [
        "Insured Gender",
        "TSA Ind",
        "SS Pri Status",
        "SS Adjustment Ind",
        "Pre-Ex Outcome",
        "SS Pri Award Amt",
        "Primary Diagnosis Category",
        "Coverage Code",
        "SIC Category",
        "pd_1_cat",
        "pd_2_cat",
    ]
    settlement = pd.get_dummies(sett_mapped, columns=TO_DUMMIES)
    settlement_comp = map_diag_entities(settlement)

    # train test consistency check...

    test_X = test_train_match(loaded_template, settlement_comp)

    # dl model prediction followed by xgb...

    best_iteration = loaded_xgb.get_booster().best_ntree_limit
    p1 = loaded_dl_model.predict(test_X.values)
    p1 = posterior_correction(P1_ORIG, P1_TRAIN, p1)

    p2 = loaded_xgb.predict_proba(test_X.values, ntree_limit=best_iteration)[:, 1]

    p2 = posterior_correction(P1_ORIG, P1_TRAIN, p2)

    # final_labels = [1 if x & y else 0
    #                 for (x, y) in zip(p1 >= 0.5, p2 >= 0.5)]  # majority scoring

    # weighted averaging
    final_test_prob = 0.3 * p1.reshape(-1,) + 0.7 * p2.reshape(
        -1,
    )

    final_labels = list((final_test_prob >= 0.5).astype(int))
    test_final = settlement.loc[:, ["Claim Identifier"]].copy()
    test_final.loc[:, "probability"] = final_test_prob
    test_final.loc[[na_ind == "Y" for na_ind in na_inds], "probability"] = np.nan
    test_final.loc[:, "p_labels_corrected"] = final_labels
    test_final.loc[[na_ind == "Y" for na_ind in na_inds], "p_labels_corrected"] = np.nan
    test_final.loc[:, "p_labels_corrected"] = test_final.loc[
        :, "p_labels_corrected"
    ].astype("Int64")
    payload_data = test_final.loc[
        :, ["Claim Identifier", "probability", "p_labels_corrected"]
    ].copy()
    payload_data.columns = ["claimNumber", "predictedProbability", "predictedValue"]
    payload_data.loc[:, "p1"] = p1
    payload_data.loc[:, "p2"] = p2
    payload_data.loc[:, "na_inds"] = na_inds
    payload_data.loc[:, "batchId"] = "batchId"
    payload_data.loc[:, "useCase"] = "use_case"
    payload_data.loc[:, "clientId"] = "clientId"
    payload_data.loc[:, "insuranceType"] = "insurance_type"

    os.remove("scaler.joblib")

    return payload_data.to_json(orient="records")


print(
    predict(
        model_name="MLMR_settlement",
        artifacts={
            "dl_model": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/dl_model_2021-06-28.h5",
            "scaler": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/scaler_2021-06-28.joblib",
            "template": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/template_data_2021-06-28.csv",
            "xgb_model": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/xgb_model_2021-06-28.joblib",
        },
        inputs={
            "claim": {
                "Mental Nervous Ind": "",
                "Recovery Amt": " 0000000000000000000000.000000",
                "Modified RTW Date": None,
                "Any Occ period": "",
                "Claim Number": "GDC-46016",
                "Policy Effective Date": "01/01/2017",
                "DOT Exertion Level (Primary)": "Unknown",
                "Last Payment To Date": "08/29/2020",
                "DOT Desc": "",
                "Elimination Period": "",
                "DOT Code": "",
                "Voc Rehab Outcome": "",
                "Policy Line of Business": "LTD",
                "Expected Term Date": None,
                "Clinical End Date": None,
                "Voc Rehab Service Requested": "",
                "Policy Termination Date": None,
                "Any Occ Start Date": "03/02/2023",
                "Duration Months": "12",
                "MentalNervousDesc": "",
                "Closed Date": None,
                "Insured State": "IL",
                "SS Pri Award Amt": "-(2000-2250)",
                "Any Occ Decision Due Date": None,
                "Elimination Days": "0",
                "SS Pursue Ind": "",
                "Any Occ Ind": "",
                "Elimination Ind": "",
                "Plan Benefit Pct": "0.6667",
                "Claim Status Description": "Benefit Case Under Review",
                "Secondary Diagnosis Code": "",
                "Secondary Diagnosis Category": "",
                "Claim Identifier": "GDC-46016-03",
                "SS Pri Award Eff Date": "09/01/2020",
                "Pre-Ex Outcome": "Y",
                "Claim Status Category": "ACTIVE",
                "Primary Diagnosis Code": "J84.9",
                "Voc Rehab Status": "",
                "Claim Cause Desc": "OTHER SICKNESS",
                "Insured Salary Ind": "WEEKLY",
                "Insured Zip": "60438",
                "SIC Code": "8069",
                "First Payment From Date": "08/29/2020",
                "SS Reject Code": "",
                "Any Occ Decision Made Date": "08/13/2020",
                "SIC Category": "",
                "Insured Age at Loss": "63",
                "Received Date": "03/12/2020",
                "Secondary Diagnosis Desc": "",
                "Voc Rehab TSA Date": None,
                "TSA Ind": "N",
                "Secondary Diagnosis Category Desc": "",
                "SIC Desc": "SPECIALTY HOSPITAL EXC., PSYCHIATRIC",
                "Claim State": "IL",
                "ThirdPartyReferralDescription": "",
                "Occ Code": "",
                "Approval Date": "08/13/2020",
                "SS Awarded Date": None,
                "Primary Diagnosis Category": "Diseases of the respiratory system",
                "Taxable Pct": "0",
                "RTW Date": None,
                "Eligibility Outcome": "Approved",
                "SS Est Start Date": None,
                "SS Pri Status": "Approved",
                "Plan Duration Date": None,
                "ThirdPartyReferralIndicator": "",
                "Primary Diagnosis Desc": "Interstitial pulmonary disease, unspecified",
                "Duration Date": "08/28/2021",
                "SocialSecurityPrimaryAwardType": "Primary Social Security Retirement with Freeze",
                "Gross Benefit Ind": 12,
                "Insured Hire Date": "04/30/1985",
                "Occ Category": "",
                "SubstanceAbuseDesc": "",
                "Insured Gender": "M",
                "Any Occ Category": "Own Occ",
                "Loss Date": "03/02/2020",
                "Voc Rehab Active Status": "",
                "Coverage Code": "LTD",
                "SS Adjustment Ind": "Y",
                "SS Eligible Ind": "Y",
                "Claim Status Code": "Open",
                "originalValues": "{'values': {'BenefitNumber': 'GDC-46016-03', 'ClaimStatusCategory': 'Open', 'ClaimCauseDescription': 'Sickness', 'InsuredGender': 'Male', 'InsuredAnnualizedSalary': '38750-39000', 'GrossBenefitIndicator': 'Monthly', 'GrossBenefit': '2000-2250', 'NetBenefit': '0-250', 'EligibilityOutcome': 'Post Approval Rules', 'BenefitCaseType': 'LTD', 'CaseSize': '400-425'}}",
                "Gross Benefit": 25500.0,
                "Insured Annualized Salary": 38875.0,
                "Net Benefit": 1500.0,
                "Policy Lives": 412.0,
                "Servicing RSO": "Chicago",
                "Nurse Cert End Date": None,
            },
            "datasetId": "spr:dataset_id",
        },
    )
)
