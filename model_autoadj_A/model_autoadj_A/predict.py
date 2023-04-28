def predict(**kwargs):
    import pandas as pd
    import pickle
    import json
    import functools

    cat_vars = [
        "Insured Gender",
        "Primary Diagnosis Category",
        "Occ Category",
        "SIC Category",
    ]

    def to_date(df, column_list):
        for col in column_list:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")
        return df

    def to_int(x, default_value=0):
        try:
            y = str(x).strip()
            return int(y) if y.isdigit() else default_value
        except Exception:
            return default_value

    def remove_comma(x):
        return int(float(str(x).replace(",", ""))) if x else None

    def fix_dates(ddf):
        ddf = to_date(
            ddf,
            [
                "Policy Effective Date",
                "Loss Date",
                "Policy Termination Date",
                "Approval Date",
                "Last Payment To Date",
                "First Payment From Date",
                "Closed Date",
                "RTW Date",
                "Received Date",
            ],
        )
        return ddf

    def set_type_age(ddf):
        ddf["Insured Age at Loss"] = ddf["Insured Age at Loss"].fillna(0).astype(int)
        return ddf

    def filter_data_bank_1(ddf):
        ddf = ddf[ddf["Claim Status Code"] != 92]
        ddf = ddf[
            (ddf["Policy Effective Date"].isnull() | ddf["Loss Date"].isnull())
            | (ddf["Loss Date"] >= ddf["Policy Effective Date"])
        ]
        ddf = ddf[
            (ddf["Policy Termination Date"].isnull() | ddf["Loss Date"].isnull())
            | (ddf["Loss Date"] <= ddf["Policy Termination Date"])
        ]
        ddf = ddf[
            (ddf["Insured Age at Loss"] >= 16) & (ddf["Insured Age at Loss"] <= 90)
        ]
        ddf = ddf[
            (ddf["Loss Date"].isnull() | ddf["Approval Date"].isnull())
            | (ddf["Loss Date"] <= ddf["Approval Date"])
        ]
        ddf = ddf[
            (
                ddf["Last Payment To Date"].isnull()
                | ddf["First Payment From Date"].isnull()
            )
            | (ddf["Last Payment To Date"] >= ddf["First Payment From Date"])
        ]
        ddf = ddf[
            (ddf["Loss Date"].isnull() | ddf["Closed Date"].isnull())
            | (ddf["Loss Date"] <= ddf["Closed Date"])
        ]
        ddf = ddf.fillna({"Policy Lives": -1})
        ddf["Policy Lives"] = ddf["Policy Lives"].apply(remove_comma)
        ddf = ddf[ddf["Policy Lives"] > 0]
        ddf = ddf[
            (
                (ddf["RTW Date"].isnull() | ddf["Loss Date"].isnull())
                | (ddf["RTW Date"] >= ddf["Loss Date"])
            )
        ]
        return ddf

    def filter_data_bank_2(ddf):
        ddf = ddf.dropna(how="any", subset=["Claim Identifier"])
        ddf = ddf[ddf["First Payment From Date"] != ""]
        ddf = ddf[~(ddf["Loss Date"] > ddf["First Payment From Date"])]
        return ddf

    def filter_data_diag_code(ddf):
        exclude_disgnosis_code_list = [
            "G43",
            "G43.0",
            "G43.001",
            "G43.909",
            "650",
            "650A",
            "650a",
        ]
        ddf = ddf[~ddf["Primary Diagnosis Code"].isin(exclude_disgnosis_code_list)]
        ddf = ddf[
            (ddf["Primary Diagnosis Code"] != "O80")
            & (ddf["Primary Diagnosis Code"] != "650")
        ]
        return ddf

    def filter_data_model_specific(ddf):
        ddf = ddf[
            (ddf["Primary Diagnosis Category"] != "Unknown")
            | (ddf["Primary Diagnosis Category"] != "Unspecified")
        ]
        ddf = ddf[ddf["SIC Category"] != "Unknown"]
        ddf = ddf[ddf["Insured Gender"] != ""]
        ddf = ddf[ddf["Occ Category"] != ""]
        ddf = ddf.dropna(subset=["Insured Age at Loss"])
        ddf = ddf.dropna(
            how="any", subset=["Insured Gender", "Occ Category", "Insured Age at Loss"]
        )
        return ddf

    def test_train_match(train_template, pred_data):
        missing_cols = set(train_template) - set(pred_data.columns)
        for c in missing_cols:
            pred_data[c] = 0
        pred_data = pred_data[train_template].copy()
        return pred_data

    def _compose2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    def _compose(*fs):
        return functools.reduce(_compose2, fs)

    data = pd.DataFrame([kwargs.get("inputs").get("claim")])
    data.loc[:, "Primary Diagnosis Category"] = data[
        "Primary Diagnosis Category"
    ].str.upper()

    filter_data_fns = _compose(
        fix_dates,
        set_type_age,
        filter_data_bank_1,
        filter_data_bank_2,
        filter_data_diag_code,
        filter_data_model_specific,
    )
    data = filter_data_fns(data)

    prediction_df = data[
        [
            "Insured Gender",
            "Insured Age at Loss",
            "Primary Diagnosis Category",
            "Occ Category",
            "SIC Category",
        ]
    ]
    prediction_df.loc[:, cat_vars] = prediction_df[cat_vars].astype("category")
    prediction_df = pd.get_dummies(
        prediction_df[
            [
                "Insured Gender",
                "Insured Age at Loss",
                "Primary Diagnosis Category",
                "Occ Category",
                "SIC Category",
            ]
        ]
    )

    with open("./model_a_artifact.pkl", "rb") as f:
        xgb_cl, cols_list = pickle.load(f)

    prediction_df = test_train_match(cols_list, prediction_df)
    prediction_df = prediction_df.drop(["class"], axis=1)
    data.loc[:, "prediction"] = xgb_cl.predict(prediction_df)

    pred_payload = data[["Claim Number", "prediction"]]
    payload_json = json.loads(pred_payload.to_json(orient="records"))[0]
    claim_number = payload_json["Claim Number"]
    return [
        {
            "inputDataSource": f"{claim_number}:0",
            "entityId": claim_number,
            "predictedResult": [
                {
                    "claim Number": claim_number,
                    "prediction": payload_json["prediction"],
                }
            ],
        }
    ]


# print(
#     predict(
#         model_name="model_autoadj",
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
#                 "Primary Diagnosis Code": "D86.85",
#                 "Voc Rehab Status": None,
#                 "Claim Cause Desc": "OTHER ACCIDENT",
#                 "Insured Salary Ind": "BI-WEEKLY",
#                 "Insured Zip": "93447",
#                 "SIC Code": 7349,
#                 "First Payment From Date": "01/28/2021",
#                 "SS Reject Code": None,
#                 "Any Occ Decision Made Date": None,
#                 "SIC Category": "Public Administration",
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
#                 "Primary Diagnosis Category": "Diseases Of The Blood",
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
#                 "Occ Category": "Officials and Managers",
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
