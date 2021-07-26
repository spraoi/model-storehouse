def predict(**kwargs):
    import pandas as pd
    import numpy as np
    import os
    import json

    from model_settlement.helpers import (
        pre_process_data,
        add_policy_tenure_to_df,
        get_date_diff,
        add_prognosis_days_to_df,
        scale_features,
        get_na_rows,
        map_categories,
        map_diag_entities,
        test_train_match,
        get_bucket_and_key_from_s3_uri,
        download_obj_from_s3,
        posterior_correction,
    )

    P1_ORIG = 0.1
    P1_TRAIN = 0.5

    for artifact in kwargs.get("artifact"):
        if artifact.get("dataName") == "dl_model_artifact":
            model_bucket, dl_model_key = get_bucket_and_key_from_s3_uri(
                artifact.get("dataValue")
            )
        elif artifact.get("dataName") == "xgb_model_artifact":
            xgb_model_key = get_bucket_and_key_from_s3_uri(artifact.get("dataValue"))[1]
        elif artifact.get("dataName") == "scaler_artifact":
            scaler_key = get_bucket_and_key_from_s3_uri(artifact.get("dataValue"))[1]
        elif artifact.get("dataName") == "template_artifact":
            template_key = get_bucket_and_key_from_s3_uri(artifact.get("dataValue"))[1]


    loaded_dl_model = download_obj_from_s3(model_bucket, dl_model_key, "dl_model")

    loaded_scaler = download_obj_from_s3(model_bucket, scaler_key, "scaler")

    loaded_xgb = download_obj_from_s3(model_bucket, xgb_model_key, "xgb_model")

    loaded_template = pd.read_csv(f"s3://{model_bucket}/{template_key}")

    data = pd.DataFrame([kwargs.get("inputs").get("claim")])

    data = data.replace(
        r"^\s*$", np.nan, regex=True
    )  # for replacing empty strings with nans ""


    settlement = pre_process_data(data)
    settlement = add_policy_tenure_to_df(settlement)
    settlement["days_to_report"] = get_date_diff(settlement["Loss Date"], settlement["Received Date"], interval="D")
    settlement["emp_tenure"] = get_date_diff(settlement["Insured Hire Date"], settlement["Loss Date"], interval="D")
    settlement = add_prognosis_days_to_df(settlement)
    settlement["days_to_first_payment"] = get_date_diff(settlement["Loss Date"], settlement["First Payment From Date"], interval="D")
    settlement = settlement.select_dtypes(exclude=["datetime64"])
    settlement = scale_features(settlement, loaded_scaler)
    settlement.drop(["Duration Months"], axis=1, inplace=True)
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
    p1 = posterior_correction(
        P1_ORIG, P1_TRAIN, p1
    )  # applying probability correction for dnn preds

    p2 = loaded_xgb.predict_proba(test_X.values, ntree_limit=best_iteration)[:, 1]
    p2 = posterior_correction(
        P1_ORIG, P1_TRAIN, p2
    )  # applying probability correction for xgb preds

    # weighted averaging preferred over majority scoring currently
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
    # removing the artifact that only works after downloading to local file system
    os.remove("scaler.joblib")

    prediction_json = json.loads(payload_data.to_json(orient="records"))
    predicted_claim = prediction_json[0] if prediction_json else None
    return [
        {"inputDataSource": f"{predicted_claim.get('claimNumber')}:0", "entityId": predicted_claim.get("claimNumber"),
         "predictedResult": predicted_claim}]


# print(
#     predict(
#         model_name="MLMR_settlement",
#         artifact=[
#         {
#             "dataId": "a117374f-fda2-44af-9e91-899e2b03b6d6",
#             "dataName": "schema_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/train_schema_2021-06-28.pbtxt",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "3cd7b112-4a7b-46f4-ad6f-aedff8e6fd4c",
#             "dataName": "statistics_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/train_stats_2021-06-28.pbtxt",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "c0c3639b-8c42-4f33-b0cc-1e3175c1254e",
#             "dataName": "feat_imp_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/feature_importances_2021-06-28.csv",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "3015dbf4-4371-4c65-80a0-abaf8d4c4372",
#             "dataName": "template_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/template_data_2021-06-28.csv",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "004f5b93-e334-4bff-865d-eb9fc487034d",
#             "dataName": "scaler_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/scaler_2021-06-28.joblib",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "68346e87-65e4-46d0-8a0e-941a0ebcc37a",
#             "dataName": "xgb_model_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/xgb_model_2021-06-28.joblib",
#             "dataValueType": "str"
#         },
#         {
#             "dataId": "1007bf57-66fc-4f19-bc1b-fce62f5e34b0",
#             "dataName": "dl_model_artifact",
#             "dataType": "artifact",
#             "dataValue": "s3://spr-ml-artifacts/prod/MLMR_settlement/artifacts/dl_model_2021-06-28.h5",
#             "dataValueType": "str"
#         }
#     ],
#         inputs={
#             "claim": {'Mental Nervous Ind': None,
#   'Recovery Amt': ' 0000000000000000005521.360000',
#   'Modified RTW Date': None,
#   'Any Occ period': None,
#   'Claim Number': 'GDC-9366',
#   'Policy Effective Date': '06/01/2014',
#   'DOT Exertion Level (Primary)': 'Unknown',
#   'Last Payment To Date': '01/17/2016',
#   'DOT Desc': None,
#   'Elimination Period': None,
#   'DOT Code': None,
#   'Voc Rehab Outcome': None,
#   'Policy Line of Business': 'LTD',
#   'Expected Term Date': None,
#   'Clinical End Date': None,
#   'Voc Rehab Service Requested': None,
#   'Policy Termination Date': None,
#   'Any Occ Start Date': None,
#   'Duration Months': '65',
#   'MentalNervousDesc': None,
#   'Closed Date': None,
#   'Insured State': 'TX',
#   'SS Pri Award Amt': '-(1250-1500)',
#   'Any Occ Decision Due Date': None,
#   'Elimination Days': None,
#   'SS Pursue Ind': None,
#   'Any Occ Ind': None,
#   'Elimination Ind': None,
#   'Plan Benefit Pct': None,
#   'Claim Status Description': 'Benefit Case Under Review',
#   'Secondary Diagnosis Code': None,
#   'Secondary Diagnosis Category': None,
#   'Claim Identifier': 'GDC-9366-01',
#   'SS Pri Award Eff Date': '08/01/2018',
#   'Pre-Ex Outcome': 'N',
#   'Claim Status Category': 'ACTIVE',
#   'Primary Diagnosis Code': 'S92.009A',
#   'Voc Rehab Status': None,
#   'Claim Cause Desc': 'OTHER ACCIDENT',
#   'Insured Salary Ind': 'MONTHLY',
#   'Insured Zip': '75803',
#   'SIC Code': '5961',
#   'First Payment From Date': '01/17/2016',
#   'SS Reject Code': None,
#   'Any Occ Decision Made Date': '12/17/2018',
#   'SIC Category': None,
#   'Insured Age at Loss': '56',
#   'Received Date': '01/12/2016',
#   'Secondary Diagnosis Desc': None,
#   'Voc Rehab TSA Date': None,
#   'TSA Ind': 'N',
#   'Secondary Diagnosis Category Desc': None,
#   'SIC Desc': 'CATALOG AND MAIL-ORDER HOUSES',
#   'Claim State': 'TX',
#   'ThirdPartyReferralDescription': None,
#   'Occ Code': None,
#   'Approval Date': '12/17/2018',
#   'SS Awarded Date': None,
#   'Primary Diagnosis Category': 'Injury, poisoning & certain other consequences of',
#   'Taxable Pct': None,
#   'RTW Date': None,
#   'Eligibility Outcome': 'Approved',
#   'SS Est Start Date': None,
#   'SS Pri Status': 'Approved',
#   'Plan Duration Date': None,
#   'ThirdPartyReferralIndicator': None,
#   'Primary Diagnosis Desc': 'Unspecified fracture of unspecified calcaneus, ini',
#   'Duration Date': '07/02/2026',
#   'SocialSecurityPrimaryAwardType': 'Primary Disability Social Security with Freeze',
#   'Gross Benefit Ind': None,
#   'Insured Hire Date': '12/23/2013',
#   'Occ Category': None,
#   'SubstanceAbuseDesc': None,
#   'Insured Gender': 'F',
#   'Any Occ Category': 'Own Occ',
#   'Loss Date': '10/19/2015',
#   'Voc Rehab Active Status': None,
#   'Coverage Code': 'LTDVOL',
#   'SS Adjustment Ind': 'Y',
#   'SS Eligible Ind': 'Y',
#   'Claim Status Code': 'Open',
#   'originalValues': "{'values': {'BenefitNumber': 'GDC-9366-01', 'ClaimStatusCategory': 'Open', 'ClaimCauseDescription': 'Accident', 'InsuredGender': 'Female', 'InsuredAnnualizedSalary': '31500-31750', 'GrossBenefitIndicator': '', 'GrossBenefit': '1500-1750', 'NetBenefit': '0-250', 'EligibilityOutcome': 'Post Approval Rules', 'BenefitCaseType': 'LTD', 'CaseSize': '900-925'}}",
#   'Gross Benefit': None,
#   'Insured Annualized Salary': 31625.0,
#   'Net Benefit': None,
#   'Policy Lives': 912.0,
#   'Servicing RSO': 'Chicago',
#   'Nurse Cert End Date': None},
#             "datasetId": "spr:dataset_id",
#         },
#     )
# )


