def predict(**kwargs):
    import pandas as pd
    import numpy as np
    from model_bridging.helpers import (
        tokenize_pd_code,
        get_date_diff,
        add_emp_tenure_to_df,
        get_bucket_and_key_from_s3_uri,
        download_model_from_s3
    )
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import WhitespaceTokenizer
    import json
    import nltk
    nltk.download("stopwords")

    def stem_text(text):
        return " ".join([stemmer.stem(w) for w in w_tokenizer.tokenize(text)])

    COLS_REQD = [
        "ClaimNumber",
        "PrimaryDiagnosisCode",
        "SICCode",
        "InsuredGender",
        "InsuredSalaryIndicator",
        "DOTPrimaryExertionLevel",
        "CaseSize",
        "PrimaryDiagnosisDecription",
        "PrimaryDiagnosisCategory",
        "InsuredAgeatLoss",
        "InsuredAnnualizedSalary",
        "InsuredHireDate",
        "ReceivedDate",
        "LossDate",
    ]
    CATEGORICAL = [
        "pd_code_1",
        "pd_code_2",
        "SIC_category",
        "InsuredGender",
        "InsuredSalaryIndicator",
        "DOTPrimaryExertionLevel",
        "CaseSize",
        "PrimaryDiagnosisCategory",
    ]
    VALID_CLAIM_STATUS_DESC = ["Benefit Case Under Review"]

    for artifact in kwargs.get("artifact"):
        if artifact.get("dataName") == "combined_artifacts":
            model_bucket, model_key = get_bucket_and_key_from_s3_uri(artifact.get("dataValue"))

    artifacts = download_model_from_s3(model_bucket, model_key)



    # Unpacking artifacts from the joblib object
    model = artifacts.get("model")
    tfidf_model = artifacts.get("tfidf_model")
    categorical_grouper = artifacts.get("categorical_grouper")
    train_template = artifacts.get("train_template")


    input_data = pd.DataFrame([kwargs.get("inputs").get("claim")])


    date_cols = [x for x in input_data.columns if "date" in x.lower()]
    for col in date_cols:
        input_data.loc[:, col] = pd.to_datetime(input_data[col], errors="coerce")

    prediction_df = input_data[
        input_data["ClaimStatusDescription"].isin(VALID_CLAIM_STATUS_DESC)
    ].copy()
    prediction_df = prediction_df.loc[
        ~(
            (prediction_df["ClaimStatusDescription"] == "Benefit Case Under Review")
            & (prediction_df["ClaimStatusCode"] == "Closed")
        ),
        :,
    ].copy()
    pred_features = prediction_df[COLS_REQD].copy().drop_duplicates()

    # tabular data preprocessing part 1
    # Extract first 2 characters from SIC code
    pred_features.loc[:, "SIC_category"] = (
        pred_features["SICCode"].astype(str).str[:2]
    )  # sic category feature
    # split primary diagnosis code into two sub-codes
    pred_features = tokenize_pd_code(pred_features)  # features from PD code
    # calculate employment tenure feature
    pred_features = add_emp_tenure_to_df(pred_features)  # emp tenure feature
    # string salary range to number conversion
    pred_features.loc[:, "InsuredAnnualizedSalary"] = [
        (float(op[0]) + float(op[1])) / 2
        for op in pred_features["InsuredAnnualizedSalary"].fillna("0-0").str.split("-")
    ]  # salary feature
    # pivot operation around Claim Number and Approval date to get sequential info in single row
    prediction_df["approval_date_rank"] = (
        prediction_df.groupby("ClaimNumber")["ApprovalDate"]
        .rank(ascending=True)
        .fillna(-1)
        .astype(int)
    )
    # get the values from the earliest snapshot
    pivot_df = prediction_df.loc[
        prediction_df.groupby("ClaimNumber").approval_date_rank.idxmin(),
        ["ClaimNumber", "BenefitCaseType", "DurationDate"],
    ].copy()
    pivot_df.rename({"DurationDate": "first_duration_date"}, axis=1, inplace=True)
    pivot_df = pivot_df.loc[pivot_df["BenefitCaseType"] == "STD"]
    # extract features for prediction
    pred_features = pivot_df.merge(pred_features, how="inner", on="ClaimNumber")
    # initial prognosis days feature
    pred_features.loc[:, "initial_prognosis_days"] = get_date_diff(
        pred_features["LossDate"], pred_features["first_duration_date"], "D"
    )
    pred_features.loc[
        pred_features["initial_prognosis_days"] <= 0, "initial_prognosis_days"
    ] = np.nan

    # text preprocessing pipeline for extracting features from
    # Primary Diagnosis Desc feature
    # initialize tokenizer, stemmer and stopwords from NLTK
    w_tokenizer = WhitespaceTokenizer()
    # lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer(language="english")
    stop = stopwords.words("english")

    # stop word removal and clean up
    pred_features.loc[:, "PrimaryDiagnosisDecription"] = (
        pred_features["PrimaryDiagnosisDecription"]
        .fillna("_na_")
        .apply(
            lambda x: " ".join([word for word in x.split(" ") if word not in (stop)])
        )
    )
    pred_features.loc[:, "PrimaryDiagnosisDecription"] = pred_features[
        "PrimaryDiagnosisDecription"
    ].str.replace("[^\w\s]", "")
    # stemming the cleaned text
    pred_features.loc[:, "pd_desc_stemmed"] = pred_features[
        "PrimaryDiagnosisDecription"
    ].apply(stem_text)
    # feature extraction from tf-idf vectorizer
    vocab = tfidf_model.get_feature_names()
    pred_desc_feat = tfidf_model.transform(
        pred_features.loc[:, "pd_desc_stemmed"]
    ).toarray()
    pred_desc_feat = pd.DataFrame(pred_desc_feat, columns=vocab)

    # adding text features to the tabular data
    x_pred = pd.concat([pred_features, pred_desc_feat], axis=1)
    # preserving training dataset feature ordering
    x_pred_sub = x_pred[train_template.columns].copy()
    x_pred_sub[CATEGORICAL] = x_pred_sub[CATEGORICAL].copy().astype(str)
    x_pred_sub[CATEGORICAL] = categorical_grouper.transform(
        x_pred_sub[CATEGORICAL].copy(), CATEGORICAL
    )
    pred_features.loc[:, "predicted_probability"] = model.predict_proba(x_pred_sub)[
        :, 1
    ]
    pred_features.loc[:, "predicted_bridged_ind"] = model.predict(x_pred_sub)

    pred_payload = pred_features[
        ["ClaimNumber", "predicted_probability", "predicted_bridged_ind"]
    ]
    payload_json = json.loads(pred_payload.to_json(orient="records"))[0]
    claim_number = payload_json["ClaimNumber"]
    return [
        {"inputDataSource": f"{claim_number}:0", "entityId": claim_number,
         "predictedResult": [{'claimNumber':claim_number,'predictedProbability':payload_json['predicted_probability'],'predicted_bridged_ind':payload_json['predicted_bridged_ind']}]}]


# print(
#     predict(
#         model_name="STD_LTD_Bridging_Model",
#         artifact= [
#     {
#         "dataId": "55dcc659-d0c5-42aa-b9bf-a0325a2997b9",
#         "dataName": "combined_artifacts",
#         "dataType": "artifact",
#         "dataValue": "s3://spr-ml-artifacts/dev/MLMR_Bridging/artifacts/all_artifacts_10-05-2021.joblib",
#         "dataValueType": "str"
#     }
# ],
#         inputs={
#             "claim":
#                 {
#                     "ClaimNumber": "GDC-13337",
#                     "BenefitNumber": "GDC-13337-01",
#                     "BenefitCaseType": "STD",
#                     "CoverageGroupCode": "STD",
#                     "ReceivedDate": "2018-11-29 15:10:16",
#                     "LossDate": "2018-11-29 00:00:00",
#                     "ClaimStatusCategory": "Open",
#                     "ClaimStatusCode": "Open",
#                     "ClaimStatusDescription": "Benefit Case Under Review",
#                     "ClaimState": "IL",
#                     "ClosedDate": None,
#                     "ClaimCauseDescription": "Sickness",
#                     "InsuredGender": "Male",
#                     "InsuredAgeatLoss": 59.0,
#                     "InsuredState": "IL",
#                     "InsuredSalaryIndicator": "MONTHLY",
#                     "InsuredAnnualizedSalary": "67500-67750",
#                     "InsuredHireDate": "2014-08-25 00:00:00",
#                     "InsuredPriAddrZipCode": "60068",
#                     "PrimaryDiagnosisCode": "M51.27",
#                     "PrimaryDiagnosisDecription": "Other intervertebral disc displacement, lumbosacra",
#                     "PrimaryDiagnosisCategory": "Diseases of the musculoskeletal system & connectiv",
#                     "SecondaryDiagnosisCode": None,
#                     "SecondaryDiagnosisDecription": None,
#                     "SecondaryDiagnosisCategory": None,
#                     "SecondaryDiagnosisCategoryDescription": None,
#                     "MentalNervousIndicator": None,
#                     "MentalNervousDescription": None,
#                     "SubstanceAbuseDescription": None,
#                     "GrossBenefitIndicator": "Weekly",
#                     "GrossBenefit": "500-750",
#                     "NetBenefit": "500-750",
#                     "TaxablePercentage": 0.0,
#                     "ReturntoWorkDate": None,
#                     "FirstPaymentfromDate": "2018-12-13 00:00:00",
#                     "LastPaymentToDate": "2018-12-13 00:00:00",
#                     "RecoveryAmount": 0.0,
#                     "ApprovalDate": "2018-12-20 14:41:19",
#                     "DurationDate": "2019-02-27 00:00:00",
#                     "DurationMonths": 33.0,
#                     "ModifiedReturntoWorkDate": None,
#                     "ExpectedTermDate": None,
#                     "ClinicalEndDate": None,
#                     "EligibilityOutcome": "Closed - Approved",
#                     "ThirdPartyReferralIndicator": None,
#                     "ThirdPartyReferralDescription": None,
#                     "EliminationPeriod": None,
#                     "EliminationIndicator": None,
#                     "EliminationDays": 0.0,
#                     "PlanDurationDate": None,
#                     "PlanBenefitPercentage": 0.6667,
#                     "SICCategory": None,
#                     "SICCode": 3451.0,
#                     "SICDesciption": "SCREW MACHINE PRODUCTS",
#                     "OccupationCategory": None,
#                     "OccupationCode": None,
#                     "DOTCode": None,
#                     "DOTDescription": None,
#                     "DOTPrimaryExertionLevel": "Light",
#                     "PolicyEffectiveDate": "2017-11-01 00:00:00",
#                     "PolicyTerminationDate": None,
#                     "CaseSize": "175-200",
#                     "AnyOccupationCategory": "Own Occ",
#                     "AnyOccupationIndicator": None,
#                     "AnyOccupationPeriod": None,
#                     "AnyOccupationStartDate": None,
#                     "AnyOccupationDecisionDueDate": None,
#                     "AnyOccupationDecisionMadeDate": None,
#                     "SocialSecurityEligibleIndicator": "Y",
#                     "SocialSecurityPrimaryPursueInd": None,
#                     "SocialSecurityPrimaryStatus": "Approved",
#                     "SocialSecurityPrimaryAwardAmt": None,
#                     "SocialSecurityPrimaryAwardType": None,
#                     "SocialSecurityPrimaryAwardEffDate": None,
#                     "SocialSecurityAdjustmentIndicator": "N",
#                     "SocialSecurityRejectCode": None,
#                     "SSEstimatedStartDate": None,
#                     "SSAwardedDate": None,
#                     "VocRehabRehabStatus": None,
#                     "VocRehabServiceRequested": None,
#                     "VocRehabActiveStatus": None,
#                     "VocRehabOutcome": None,
#                     "PreExistingOutcome": "Y",
#                     "TSAIndicator": "N",
#                     "TSADate": None
#                 }
#         },
#     )
# )
