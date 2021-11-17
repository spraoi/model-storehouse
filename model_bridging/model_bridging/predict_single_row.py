def predict_local(**kwargs):
    # loading local artifacts currently
    # same artifact can be found here "s3://spr-ml-artifacts/dev/MLMR_Bridging/artifacts/all_artifacts_10-05-2021.joblib"
    import joblib
    import pandas as pd
    import numpy as np
    from model_bridging.helpers import (
        tokenize_pd_code,
        get_date_diff,
        add_emp_tenure_to_df,
    )
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import WhitespaceTokenizer
    import json

    def stem_text(text):
        return " ".join([stemmer.stem(w) for w in w_tokenizer.tokenize(text)])

    # ARTIFACT_PATH = "artifacts/all_artifacts_10-05-2021.joblib"
    ARTIFACT_PATH = "s3://spr-ml-artifacts/dev/MLMR_Bridging/artifacts/all_artifacts_10-05-2021.joblib"
    JSON_PATH = kwargs.get("inputs").get("input_json")  # "single_row_bridging.json"

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
    artifacts = joblib.load(ARTIFACT_PATH)
    # Unpacking artifacts from the joblib object
    model = artifacts.get("model")
    tfidf_model = artifacts.get("tfidf_model")
    categorical_grouper = artifacts.get("categorical_grouper")
    train_template = artifacts.get("train_template")

    print(f"train template columns: \n {train_template.columns[:12]}")

    dump_2021 = pd.read_json(JSON_PATH)

    print(f"input columns length: {dump_2021.shape[1]}")

    date_cols = [x for x in dump_2021.columns if "date" in x.lower()]
    for col in date_cols:
        dump_2021.loc[:, col] = pd.to_datetime(dump_2021[col], errors="coerce")

    prediction_df = dump_2021[
        dump_2021["ClaimStatusDescription"].isin(VALID_CLAIM_STATUS_DESC)
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
    ].copy()
    payload_json = json.loads(pred_payload.to_json(orient="records"))
    return payload_json


if __name__ == "__main__":
    payload = predict_local(
        inputs={
            "modelId": "spr:bz:mod::487308e2-a9d2-4e93-9b01-1303f195aab5",
            "version_number": 1,
            "input_json": "input_json_2.json",
        }
    )
    print(payload)
