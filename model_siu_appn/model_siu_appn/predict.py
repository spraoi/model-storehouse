def predict(**kwargs):
    import pandas as pd
    import tempfile
    import boto3
    import pickle
    import json
    import warnings

    warnings.filterwarnings("ignore")

    int_cols = [
        "FINANCIALGOALSDATA_SUITABILITYSCORE",
        "uc2_employee_flag",
        "uc2_employee_total_policy_count",
        "uc2_employee_policy_value",
        "uc7_employee_ratio",
        "uc8_employee_flag",
        "POLICYCOVHDRDATA_ISSUEAGE",
        "NBPAYMENTSDATA_FORMATAMOUNT",
        "NBAGENTDATA_PERCENTAGE",
        "NBANNUITYDATA_CHARGESINCURREPLACE",
        "NBANNUITYDATA_CASHWITHAPP",
        "NBPAYMENTSDATA_COMMISSIONRETAINED",
        "NBREPLACEMENTINFODATA_ESTIMATEDVALUE",
        "NBANNUITYDATA_EXPECTEDPREM",
        "TRXHDRDATA_PROCESSEDDATEdiff",
        "POLICYCOVHDRDATA_ISSUEDATEdiff",
        "NBANNUITYDATA_APPSIGNDATEdiff",
        "NBREPLACEMENTINFODATA_PAPERWORKRECEIVEDDATEdiff",
        "APPBENEFITDATA_STARTDATEdiff",
        "CASEHDRDATA_DATEISSUEDdiff",
        "CASEHDRDATA_DATERECEIVEDdiff",
        "CASEHDRDATA_EAPPDATARECEIVEDdiff",
        "CASEHDRDATA_TIMESTAMPdiff",
        "FINANCIALGOALSDATA_TIMESTAMPdiff",
        "no_employees",
        "NBANNUITYDATA_ANNUITYPOLICYOPTION_SPDA",
        "webservices",
    ]

    cat_cols = [
        "NBANNUITYDATA_APPSIGNSTATE_VALUE",
        "POLICYHDRDATA_PRODMODELID_VALUE",
        "NBANNUITYDATA_APPSTATUS_VALUE",
        "NBANNUITYDATA_APPSOURCE_VALUE",
        "APPCOVERAGESDATA_PRODMODELID_VALUE",
        "NBANNUITYDATA_OWNERPARTYTYPE_VALUE",
        "NBANNUITYDATA_ESIGNATUREIND_VALUE",
        "NBANNUITYDATA_OWNERANNUITANTFLAG_VALUE",
        "NBANNUITYDATA_ISREPLACEMENT_VALUE",
        "NBANNUITYDATA_ISGROUP_VALUE",
        "NBPAYMENTSDATA_PREMIUMTYPE_CODE",
        "NBANNUITYDATA_ANNUITYPOLICYOPTION",
        "NBANNUITYDATA_COMMOPTIONTYPE_VALUE",
        "NBANNUITYDATA_ANNUITYPOLICYTYPE",
        "NBANNUITYDATA_ADEQUATEASSETS_VALUE",
        "NBANNUITYDATA_PURPOSEOFACCOUNT_VALUE",
        "NBPAYMENTSDATA_METHOD",
        "NBANNUITYDATA_SYSTEMATICWITHDRAWAL_VALUE",
        "NBANNUITYDATA_CHECKFORMS_VALUE",
        "NBANNUITYDATA_ROLLOVER_VALUE",
        "NBANNUITYDATA_QUALIFIED_VALUE",
        "NBAGENTDATA_ISPRIMARY_VALUE",
        "CASEHDRDATA_ISGROUP_VALUE",
        "CASEHDRDATA_STATUS_VALUE",
        "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE",
        "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE",
        "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE",
        "CASEHDRDATA_ESIGNATUREIND_VALUE",
        "FINANCIALGOALSDATA_ADEQUATEASSETS_VALUE",
    ]

    bool_cols = [
        "emp_level_1",
        "emp_level_2",
        "emp_level_3",
        "emp_level_4",
        "emp_level_5",
        "emp_level_6",
        "emp_level_7",
        "emp_level_8",
        "emp_level_9",
        "emp_level_10",
        "emp_level_11",
        "emp_level_12",
        "emp_level_13",
        "emp_level_14",
        "emp_level_15",
        "emp_level_16",
        "emp_level_17",
        "emp_level_18",
        "emp_level_19",
        "emp_level_20",
        "emp_level_21",
        "emp_level_22",
    ]

    qt = [
        "FINANCIALGOALSDATA_SUITABILITYSCORE",
        "uc2_employee_total_policy_count",
        "uc2_employee_policy_value",
        "uc7_employee_ratio",
        "POLICYCOVHDRDATA_ISSUEAGE",
        "NBPAYMENTSDATA_FORMATAMOUNT",
        "NBAGENTDATA_PERCENTAGE",
        "NBANNUITYDATA_CHARGESINCURREPLACE",
        "NBANNUITYDATA_CASHWITHAPP",
        "NBPAYMENTSDATA_COMMISSIONRETAINED",
        "NBREPLACEMENTINFODATA_ESTIMATEDVALUE",
        "NBANNUITYDATA_EXPECTEDPREM",
        "TRXHDRDATA_PROCESSEDDATEdiff",
        "POLICYCOVHDRDATA_ISSUEDATEdiff",
        "NBANNUITYDATA_APPSIGNDATEdiff",
        "NBREPLACEMENTINFODATA_PAPERWORKRECEIVEDDATEdiff",
        "APPBENEFITDATA_STARTDATEdiff",
        "CASEHDRDATA_DATEISSUEDdiff",
        "CASEHDRDATA_DATERECEIVEDdiff",
        "CASEHDRDATA_EAPPDATARECEIVEDdiff",
        "CASEHDRDATA_TIMESTAMPdiff",
        "FINANCIALGOALSDATA_TIMESTAMPdiff",
        "NBANNUITYDATA_ANNUITYPOLICYOPTION_SPDA",
    ]

    THRESHOLD = 0.5056200992793571

    def get_bucket_and_key_from_s3_uri(uri):
        bucket, key = uri.split("/", 2)[-1].split("/", 1)
        return bucket, key

    def download_model_from_s3(bucket_name, key):
        bucket = boto3.resource("s3").Bucket(bucket_name)
        with tempfile.NamedTemporaryFile() as fp:
            bucket.download_fileobj(key, fp)
            loaded_model = pickle.load(open(fp.name, "rb"))
        return loaded_model

    def preprocess_employeeid(df, emp_col="V_SYSTEM_LOGON_ID"):
        df["V_SYSTEM_LOGON_ID"] = df["V_SYSTEM_LOGON_ID"].fillna("None")

        df["webservices"] = df[emp_col].apply(lambda x: 1 if "WebServices" in x else 0)
        df["emp_level_1"] = df[emp_col].apply(lambda x: 1 if "160frx" in x else 0)
        df["emp_level_2"] = df[emp_col].apply(lambda x: 1 if "160mrx" in x else 0)
        df["emp_level_3"] = df[emp_col].apply(lambda x: 1 if "162blx" in x else 0)
        df["emp_level_4"] = df[emp_col].apply(lambda x: 1 if "470FJX" in x else 0)
        df["emp_level_5"] = df[emp_col].apply(lambda x: 1 if "470bmx" in x else 0)
        df["emp_level_6"] = df[emp_col].apply(lambda x: 1 if "470bpx" in x else 0)
        df["emp_level_7"] = df[emp_col].apply(lambda x: 1 if "470ckx" in x else 0)
        df["emp_level_8"] = df[emp_col].apply(lambda x: 1 if "470ddx" in x else 0)
        df["emp_level_9"] = df[emp_col].apply(lambda x: 1 if "470djl" in x else 0)
        df["emp_level_10"] = df[emp_col].apply(lambda x: 1 if "470fnx" in x else 0)
        df["emp_level_11"] = df[emp_col].apply(lambda x: 1 if "470fsx" in x else 0)
        df["emp_level_12"] = df[emp_col].apply(lambda x: 1 if "470grw" in x else 0)
        df["emp_level_13"] = df[emp_col].apply(lambda x: 1 if "470kax" in x else 0)
        df["emp_level_14"] = df[emp_col].apply(lambda x: 1 if "470kse" in x else 0)
        df["emp_level_15"] = df[emp_col].apply(lambda x: 1 if "470lcr" in x else 0)
        df["emp_level_16"] = df[emp_col].apply(lambda x: 1 if "470lgk" in x else 0)
        df["emp_level_17"] = df[emp_col].apply(lambda x: 1 if "470max" in x else 0)
        df["emp_level_18"] = df[emp_col].apply(lambda x: 1 if "470sce" in x else 0)
        df["emp_level_19"] = df[emp_col].apply(lambda x: 1 if "470tas" in x else 0)
        df["emp_level_20"] = df[emp_col].apply(lambda x: 1 if "470vrd" in x else 0)
        df["emp_level_21"] = df[emp_col].apply(lambda x: 1 if "470wbm" in x else 0)
        df["emp_level_22"] = df[emp_col].apply(lambda x: 1 if "725ccy" in x else 0)

        df["no_employees"] = df[emp_col].apply(lambda x: len(x.split(",")))

        emp_level_cols = [
            "emp_level_1",
            "emp_level_2",
            "emp_level_3",
            "emp_level_4",
            "emp_level_5",
            "emp_level_6",
            "emp_level_7",
            "emp_level_8",
            "emp_level_9",
            "emp_level_10",
            "emp_level_11",
            "emp_level_12",
            "emp_level_13",
            "emp_level_14",
            "emp_level_15",
            "emp_level_16",
            "emp_level_17",
            "emp_level_18",
            "emp_level_19",
            "emp_level_20",
            "emp_level_21",
            "emp_level_22",
        ]

        df[["webservices", "no_employees"]] = df[["webservices", "no_employees"]].apply(
            pd.to_numeric, errors="coerce"
        )
        df[emp_level_cols] = df[emp_level_cols].astype("bool")
        df = df.drop(emp_col, axis=1)
        return df

    # init resources
    for artifact in kwargs.get("artifact"):
        if artifact.get("dataName") == "combined_artifacts":
            model_bucket, model_key = get_bucket_and_key_from_s3_uri(
                artifact.get("dataValue")
            )

    quant_transformer, catboost_encoder, lgb_clf = download_model_from_s3(
        model_bucket, model_key
    )
    pd_df = pd.DataFrame(kwargs.get("inputs").get("claim"))

    # process_nan's
    pd_df["uc2_employee_ratio"] = pd_df["uc2_employee_ratio"].fillna(-1)
    pd_df["uc7_employee_ratio"] = pd_df["uc7_employee_ratio"].fillna(-1)
    pd_df["uc8_employee_mean_change_days"] = pd_df[
        "uc8_employee_mean_change_days"
    ].fillna(-1)
    pd_df["uc8_employee_ratio"] = pd_df["uc8_employee_ratio"].fillna(-1)

    pd_df["NBPAYMENTSDATA_PREMIUMTYPE_CODE"] = pd_df[
        "NBPAYMENTSDATA_PREMIUMTYPE_CODE"
    ].fillna("UNK")
    pd_df["NBANNUITYDATA_PURPOSEOFACCOUNT_VALUE"] = pd_df[
        "NBANNUITYDATA_PURPOSEOFACCOUNT_VALUE"
    ].fillna("UNK")
    pd_df["NBPAYMENTSDATA_METHOD"] = pd_df["NBPAYMENTSDATA_METHOD"].fillna("UNK")

    pd_df["NBAGENTDATA_ISPRIMARY_VALUE"] = pd_df["NBAGENTDATA_ISPRIMARY_VALUE"].fillna(
        "Blank"
    )
    pd_df["FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE"] = pd_df[
        "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE"
    ].fillna("Blank")
    pd_df["FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE"] = pd_df[
        "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE"
    ].fillna("Blank")

    # entities
    TXN_NUM = pd_df.pop("TRXNUM")

    # preprocess
    pd_df = preprocess_employeeid(pd_df)
    pd_df[cat_cols] = pd_df[cat_cols].astype("category")
    pd_df[bool_cols] = pd_df[bool_cols].astype("bool")
    pd_df = pd_df[int_cols + cat_cols + bool_cols]

    # Robust scaler
    pd_df.loc[:, qt] = quant_transformer.transform(pd_df.loc[:, qt].to_numpy())

    # Catboost encoding
    pd_df_enc = catboost_encoder.transform(pd_df[cat_cols])
    pd_df = pd_df.drop(columns=cat_cols, inplace=False)
    pd_df = pd.concat([pd_df, pd_df_enc], axis=1)

    # 1->inconsistent, 0->consistent
    pd_df["predicted_disposition"] = (
        lgb_clf.predict_proba(pd_df)[:, 1] >= THRESHOLD
    ).astype(int)

    # Payload gen
    pd_df["TRXNUM"] = TXN_NUM
    pd_df = pd_df[["TRXNUM", "predicted_disposition"]]
    prediction_json = json.loads(pd_df.to_json(orient="records"))
    predicted_claim = prediction_json if prediction_json else None

    return [
        {
            "inputDataSource": "RANDN1209:0",  # temporarily present for payload schema restriction
            "entityId": "RAND1209",  # temporarily present for payload schema restriction
            "predictedResult": predicted_claim,
        }
    ]


# print(
#     predict(
#         model_name="SIU_TXN_model_v5",
#         artifact=[
#             {
#                 "dataId": "55dcc659-d0c5-42aa-b9bf-a0325a2997b9",
#                 "dataName": "combined_artifacts",
#                 "dataType": "artifact",
#                 "dataValue": "s3://siutempbucket/tariq/combined_appn_siu_c_v5.sav",
#                 "dataValueType": "str",
#             }
#         ],
#         inputs={
#             "claim": {
#                 "shortId": {0: "2214952", 1: "2214975"},
#                 "feedback_field": {0: "other", 1: "other"},
#                 "feedback_reason": {0: "Missed NIGO", 1: "Issued IGO"},
#                 "POLICYNUMBER": {0: "AAMGE00237", 1: "AAMGE00240"},
#                 "FINANCIALGOALSDATA_SUITABILITYSCORE": {0: 15.0, 1: 15.0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE": {0: "NJ", 1: "NJ"},
#                 "uc5_flag": {0: 0.0, 1: 0.0},
#                 "EMPLOYEE_ID": {
#                     0: "['500901F7D9E57D2FE0531100F20A6987', '298']",
#                     1: "['500901F7D9E57D2FE0531100F20A6987', '215']",
#                 },
#                 "V_SYSTEM_LOGON_ID": {
#                     0: "['WebServices', '470ckx']",
#                     1: "['WebServices', '470sce']",
#                 },
#                 "uc2_employee_flag": {0: 0.0, 1: 0.0},
#                 "uc2_employee_total_policy_count": {0: 6255.0, 1: 5616.0},
#                 "uc2_employee_withdrawl_policy_count": {0: 0.0, 1: 0.0},
#                 "uc2_employee_policy_value": {0: 0.0, 1: 0.0},
#                 "uc2_employee_policy_withdrawl": {0: 0.0, 1: 0.0},
#                 "uc2_employee_ratio": {0: None, 1: None},
#                 "uc7_employee_flag": {0: 0.0, 1: 0.0},
#                 "uc7_employee_policy_count": {0: 6255.0, 1: 5616.0},
#                 "uc7_employee_task_count": {0: 2045.0, 1: 1603.0},
#                 "uc7_employee_high_task_count": {0: 34.0, 1: 26.0},
#                 "uc7_employee_zero_task_count": {0: 4785.0, 1: 4442.0},
#                 "uc7_employee_ratio": {0: 0.0231292517006802, 1: 0.0221465076660988},
#                 "uc8_employee_flag": {0: 1.0, 1: 0.0},
#                 "uc8_employee_policy_count": {0: 68.0, 1: 64.0},
#                 "uc8_employee_change_policy_count": {0: 1.0, 1: 0.0},
#                 "uc8_employee_mean_change_days": {0: 7.0, 1: None},
#                 "uc8_employee_party_change_count": {0: 1.0, 1: 0.0},
#                 "uc8_employee_ratio": {0: 0.0147058823529411, 1: 0.0},
#                 "numerator": {0: 4, 1: 4},
#                 "denominator": {0: 4, 1: 4},
#                 "score": {0: 0.25, 1: 0.0},
#                 "predicted_disposition": {0: "inconsistent", 1: "inconsistent"},
#                 "eligible_hypothesis": {0: "4/4", 1: "4/4"},
#                 "CONTRACTDATA_WRITINGCODE": {0: "['IND0004582']", 1: "['IND0004582']"},
#                 "TRXNUM": {0: 2214952.0, 1: 2214975.0},
#                 "TRXHDRDATA_PROCESSEDDATE": {0: "2020-10-29", 1: "2020-10-29"},
#                 "TRXHDRDATA_TRXTYPEID_VALUE": {0: "Issue", 1: "Issue"},
#                 "POLICYCOVHDRDATA_ISSUEDATE": {0: "2020-10-29", 1: "2020-10-29"},
#                 "POLICYCOVHDRDATA_ISSUEAGE": {0: 72, 1: 72},
#                 "POLICYHDRDATA_PRODMODELID_VALUE": {
#                     0: "RSL - Apollo",
#                     1: "RSL - Apollo",
#                 },
#                 "NBANNUITYDATA_APPSTATUS_VALUE": {0: "New", 1: "New"},
#                 "NBANNUITYDATA_APPDATERECEIVED": {0: "2020-09-16", 1: "2020-10-20"},
#                 "NBANNUITYDATA_APPSIGNDATE": {0: "2020-09-16", 1: "2020-10-20"},
#                 "NBANNUITYDATA_APPSOURCE_VALUE": {
#                     0: "E-App Firelight",
#                     1: "E-App Firelight",
#                 },
#                 "APPCOVERAGESDATA_PRODMODELID_VALUE": {
#                     0: "RSL - Apollo",
#                     1: "RSL - Apollo",
#                 },
#                 "NBPAYMENTSDATA_PREMIUMTYPE_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_OWNERPARTYTYPE_VALUE": {0: "Person", 1: "Person"},
#                 "NBREPLACEMENTINFODATA_PAPERWORKRECEIVEDDATE": {0: None, 1: None},
#                 "APPBENEFITDATA_STARTDATE": {0: "2020-10-29", 1: "2020-10-29"},
#                 "NBPAYMENTSDATA_FORMATAMOUNT": {0: -1.0, 1: -1.0},
#                 "NBAGENTDATA_PERCENTAGE": {0: 100.0, 1: 100.0},
#                 "NBANNUITYDATA_ESIGNATUREIND_VALUE": {0: True, 1: True},
#                 "NBANNUITYDATA_OWNERANNUITANTFLAG_VALUE": {0: True, 1: True},
#                 "NBANNUITYDATA_CHARGESINCURREPLACE": {0: 0.0, 1: 0.0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_OWNERORGTYPE_VALUE": {0: None, 1: None},
#                 "NBREPLACEMENTINFODATA_PAYMENTSOURCETYPE_CODE": {0: None, 1: None},
#                 "NBANNUITYDATA_CASHWITHAPP": {0: 0.0, 1: 0.0},
#                 "NBANNUITYDATA_PAYMENTMETHOD_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_ISREPLACEMENT_VALUE": {0: False, 1: False},
#                 "NBANNUITYDATA_ANNUITANTMARSTAT_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_ISGROUP_VALUE": {0: True, 1: True},
#                 "NBPAYMENTSDATA_COMMISSIONRETAINED": {0: -1.0, 1: -1.0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_CODE": {0: None, 1: None},
#                 "NBPAYMENTSDATA_NETCOMMISSIONS": {0: -1.0, 1: -1.0},
#                 "NBANNUITYDATA_APPTYPE_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_ANNUITYPOLICYOPTION": {0: "SPDA", 1: "SPDA"},
#                 "NBANNUITYDATA_COMMOPTIONTYPE_VALUE": {0: "Option A", 1: "Option A"},
#                 "NBANNUITYDATA_ANNUITYPOLICYTYPE": {
#                     0: "BaseCov_SPDAMVA",
#                     1: "BaseCov_SPDAMVA",
#                 },
#                 "NBANNUITYDATA_ADEQUATEASSETS_VALUE": {0: True, 1: True},
#                 "NBANNUITYDATA_PURPOSEOFACCOUNT_VALUE": {
#                     0: "Premium Billing",
#                     1: "Premium Billing",
#                 },
#                 "NBREPLACEMENTINFODATA_REPLCO1035_VALUE": {0: None, 1: None},
#                 "NBREPLACEMENTINFODATA_ESTIMATEDVALUE": {0: -1.0, 1: -1.0},
#                 "NBREPLACEMENTINFODATA_PARTIALORFULL_VALUE": {0: None, 1: None},
#                 "NBPAYMENTSDATA_METHOD": {0: None, 1: None},
#                 "NBANNUITYDATA_SYSTEMATICWITHDRAWAL_VALUE": {0: False, 1: False},
#                 "NBREPLACEMENTINFODATA_EXCHANGETYPE_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_ACCOUNTTYPE_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_CHECKFORMS_VALUE": {0: True, 1: True},
#                 "NBANNUITYDATA_ROLLOVER_VALUE": {0: False, 1: False},
#                 "NBANNUITYDATA_QUALIFIED_VALUE": {0: True, 1: False},
#                 "NBAGENTDATA_ISPRIMARY_VALUE": {0: True, 1: True},
#                 "NBREPLACEMENTINFODATA_ACCTYPE_VALUE": {0: None, 1: None},
#                 "NBPAYMENTSDATA_AMOUNT": {0: -1.0, 1: -1.0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE": {0: None, 1: None},
#                 "NBANNUITYDATA_EXPECTEDPREM": {0: 28000.0, 1: 250444.32},
#                 "NBANNUITYDATA_ANNUITANTAGE": {0: 72.0, 1: 72.0},
#                 "NBANNUITYDATA_COMMISSIONSWITHHELD": {0: 0.0, 1: 0.0},
#                 "NBANNUITYDATA_EXPECTEDSURRPENALTY_VALUE": {0: "No", 1: "No"},
#                 "CASEHDRDATA_APPSIGNDATE": {0: "2020-09-16", 1: "2020-10-20"},
#                 "CASEHDRDATA_DATEISSUED": {0: "2020-10-29", 1: "2020-10-29"},
#                 "CASEHDRDATA_DATERECEIVED": {0: "2020-09-16", 1: "2020-10-20"},
#                 "CASEHDRDATA_EAPPDATARECEIVED": {0: "2020-09-16", 1: "2020-10-20"},
#                 "CASEHDRDATA_RATELOCKEFFECTIVEDATE": {0: "2020-09-16", 1: "2020-10-29"},
#                 "CASEHDRDATA_TIMESTAMP": {
#                     0: "2020-10-29 11:13:42.935540500",
#                     1: "2020-10-29 11:56:14.714622100",
#                 },
#                 "FINANCIALGOALSDATA_TIMESTAMP": {
#                     0: "2020-09-16 19:46:04.847665300",
#                     1: "2020-10-20 19:26:36.719643600",
#                 },
#                 "CASEHDRDATA_ISGROUP_VALUE": {0: True, 1: True},
#                 "FINANCIALGOALSDATA_EXPECTEDPREM": {0: 28000.0, 1: 250444.32},
#                 "CASEHDRDATA_STATUS_VALUE": {0: "Issued", 1: "New"},
#                 "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE": {
#                     0: "Last Requirement",
#                     1: "Requirement Review",
#                 },
#                 "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE": {0: "No", 1: "No"},
#                 "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE": {0: "No", 1: "No"},
#                 "CASEHDRDATA_ESIGNATUREIND_VALUE": {0: True, 1: True},
#                 "FINANCIALGOALSDATA_ADEQUATEASSETS_VALUE": {0: True, 1: True},
#                 "TRXHDRDATA_PROCESSEDDATEdiff": {0: -1, 1: -1},
#                 "POLICYCOVHDRDATA_ISSUEDATEdiff": {0: -1, 1: -1},
#                 "NBANNUITYDATA_APPSIGNDATEdiff": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PAPERWORKRECEIVEDDATEdiff": {0: -1, 1: -1},
#                 "APPBENEFITDATA_STARTDATEdiff": {0: -1, 1: -1},
#                 "CASEHDRDATA_APPSIGNDATEdiff": {0: 0, 1: 0},
#                 "CASEHDRDATA_DATEISSUEDdiff": {0: -1, 1: -1},
#                 "CASEHDRDATA_DATERECEIVEDdiff": {0: 0, 1: 0},
#                 "CASEHDRDATA_EAPPDATARECEIVEDdiff": {0: 0, 1: 0},
#                 "CASEHDRDATA_RATELOCKEFFECTIVEDATEdiff": {0: 0, 1: -1},
#                 "CASEHDRDATA_TIMESTAMPdiff": {0: -1, 1: -1},
#                 "FINANCIALGOALSDATA_TIMESTAMPdiff": {0: 0, 1: 0},
#                 "no_employees": {0: 1, 1: 1},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_AK": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_AL": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_AR": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_AZ": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_CA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_CO": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_CT": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_DC": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_DE": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_FL": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_GA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_HI": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_IA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_ID": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_IL": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_IN": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_KS": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_KY": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_LA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_MA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_MD": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_ME": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_MI": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_MN": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_MO": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_MS": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_MT": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_NC": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_ND": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_NE": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_NH": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_NJ": {0: 1, 1: 1},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_NM": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_NV": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_OH": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_OK": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_OR": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_PA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_PR": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_RI": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_SC": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_SD": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_TN": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_TX": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_UT": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_VA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_VT": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_WA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_WI": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_WV": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSIGNSTATE_VALUE_WY": {0: 0, 1: 0},
#                 "predicted_disposition_consistent": {0: 0, 1: 1},
#                 "predicted_disposition_inconsistent": {0: 1, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Cost Basis Adjustment": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim - 5YD": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim - LD": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim Disbursement": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Free Look Cancel": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Full Surrender": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Issue": {0: 1, 1: 1},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Partial Surrender": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Spousal Continuation": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Value Adjustment": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim - 10YD": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Annuitization": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Dividend Release": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Annuity Payment": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim Annuitization": {0: 0, 1: 0},
#                 "TRXHDRDATA_TRXTYPEID_VALUE_DOB Correction": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Apollo": {0: 1, 1: 1},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Apollo 93 BV": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Apollo 93 MVA": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Argus BV": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Argus MVA": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Cornerstone": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Deferral": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Elektra 5 6 7": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Eleos": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Eleos 96 BV": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Eleos 96 MVA": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - FPDA": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - John Alden": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Keystone": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Reliance Guarantee": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "POLICYHDRDATA_PRODMODELID_VALUE_SSL - Converted": {0: 0, 1: 0},
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Reliance Accumulator": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Deferral Annuitization": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBANNUITYDATA_APPSTATUS_VALUE_New": {0: 1, 1: 1},
#                 "NBANNUITYDATA_APPSTATUS_VALUE_Submitted": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSOURCE_VALUE_E-App": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPSOURCE_VALUE_E-App Firelight": {0: 1, 1: 1},
#                 "NBANNUITYDATA_APPSOURCE_VALUE_Mail": {0: 0, 1: 0},
#                 "APPCOVERAGESDATA_PRODMODELID_VALUE_RSL - Apollo": {0: 1, 1: 1},
#                 "APPCOVERAGESDATA_PRODMODELID_VALUE_RSL - Eleos": {0: 0, 1: 0},
#                 "APPCOVERAGESDATA_PRODMODELID_VALUE_RSL - Keystone": {0: 0, 1: 0},
#                 "APPCOVERAGESDATA_PRODMODELID_VALUE_RSL - Reliance Accumulator": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "APPCOVERAGESDATA_PRODMODELID_VALUE_RSL - Reliance Guarantee": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBPAYMENTSDATA_PREMIUMTYPE_VALUE_1035 Exchange": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_VALUE_Direct Transfer": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_VALUE_Indirect Rollover": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_VALUE_New Money": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_VALUE_Direct Rollover": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERPARTYTYPE_VALUE_Organization": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERPARTYTYPE_VALUE_Person": {0: 1, 1: 1},
#                 "NBANNUITYDATA_ESIGNATUREIND_VALUE_False": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ESIGNATUREIND_VALUE_True": {0: 1, 1: 1},
#                 "NBANNUITYDATA_OWNERANNUITANTFLAG_VALUE_False": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERANNUITANTFLAG_VALUE_True": {0: 1, 1: 1},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Business": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Child": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Employer": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Other": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Self": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Spouse": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Third Party Owner": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Trust": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Unknown": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Estate": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Trustee": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Charity": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Husband": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Sister": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Applicant": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Sibling": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Business Partner": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Father": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Legal Ward": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Niece": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Son": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Testamentary Trust": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Grandson": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERRELATION_VALUE_Nephew": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERORGTYPE_VALUE_Corporation": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERORGTYPE_VALUE_Custodial": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERORGTYPE_VALUE_Other": {0: 0, 1: 0},
#                 "NBANNUITYDATA_OWNERORGTYPE_VALUE_Trust": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PAYMENTSOURCETYPE_CODE_Replacement": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PAYMENTSOURCETYPE_CODE_Rollover": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PAYMENTSOURCETYPE_CODE_Transfer": {0: 0, 1: 0},
#                 "NBANNUITYDATA_PAYMENTMETHOD_VALUE_Check": {0: 0, 1: 0},
#                 "NBANNUITYDATA_PAYMENTMETHOD_VALUE_Wire": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ISREPLACEMENT_VALUE_False": {0: 1, 1: 1},
#                 "NBANNUITYDATA_ISREPLACEMENT_VALUE_True": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITANTMARSTAT_VALUE_Divorced": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITANTMARSTAT_VALUE_Domestic Partnership": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBANNUITYDATA_ANNUITANTMARSTAT_VALUE_Married": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITANTMARSTAT_VALUE_Single": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITANTMARSTAT_VALUE_Widowed": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ISGROUP_VALUE_False": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ISGROUP_VALUE_True": {0: 1, 1: 1},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_CODE_1035EX": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_CODE_DirTran": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_CODE_IndRoll": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_CODE_NewMoney": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_PREMIUMTYPE_CODE_DirRoll": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPTYPE_VALUE_Full Application": {0: 0, 1: 0},
#                 "NBANNUITYDATA_APPTYPE_VALUE_LITE App": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITYPOLICYOPTION_SPDA": {0: 1, 1: 1},
#                 "NBANNUITYDATA_ANNUITYPOLICYOPTION_SPDA_5Yr": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITYPOLICYOPTION_SPDA_7Yr": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITYPOLICYOPTION_SPDA_10Yr": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITYPOLICYOPTION_SPDA_6Yr": {0: 0, 1: 0},
#                 "NBANNUITYDATA_COMMOPTIONTYPE_VALUE_Option A": {0: 1, 1: 1},
#                 "NBANNUITYDATA_COMMOPTIONTYPE_VALUE_Option B": {0: 0, 1: 0},
#                 "NBANNUITYDATA_COMMOPTIONTYPE_VALUE_Option C": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITYPOLICYTYPE_BaseCov_SPDA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ANNUITYPOLICYTYPE_BaseCov_SPDAMVA": {0: 1, 1: 1},
#                 "NBANNUITYDATA_ANNUITYPOLICYTYPE_BaseCov_SPEIDA": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ADEQUATEASSETS_VALUE_False": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ADEQUATEASSETS_VALUE_True": {0: 1, 1: 1},
#                 "NBANNUITYDATA_PURPOSEOFACCOUNT_VALUE_Disbursements": {0: 0, 1: 0},
#                 "NBANNUITYDATA_PURPOSEOFACCOUNT_VALUE_Premium Billing": {0: 1, 1: 1},
#                 "NBREPLACEMENTINFODATA_REPLCO1035_VALUE_False": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_REPLCO1035_VALUE_True": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PARTIALORFULL_VALUE_Full": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PARTIALORFULL_VALUE_Partial": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_METHOD_CLH": {0: 0, 1: 0},
#                 "NBPAYMENTSDATA_METHOD_EXC": {0: 0, 1: 0},
#                 "NBANNUITYDATA_SYSTEMATICWITHDRAWAL_VALUE_False": {0: 1, 1: 1},
#                 "NBANNUITYDATA_SYSTEMATICWITHDRAWAL_VALUE_True": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_EXCHANGETYPE_VALUE_External": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_EXCHANGETYPE_VALUE_Internal": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ACCOUNTTYPE_VALUE_Checking": {0: 0, 1: 0},
#                 "NBANNUITYDATA_ACCOUNTTYPE_VALUE_Savings": {0: 0, 1: 0},
#                 "NBANNUITYDATA_CHECKFORMS_VALUE_False": {0: 0, 1: 0},
#                 "NBANNUITYDATA_CHECKFORMS_VALUE_True": {0: 1, 1: 1},
#                 "NBANNUITYDATA_ROLLOVER_VALUE_False": {0: 1, 1: 1},
#                 "NBANNUITYDATA_ROLLOVER_VALUE_True": {0: 0, 1: 0},
#                 "NBANNUITYDATA_QUALIFIED_VALUE_False": {0: 0, 1: 1},
#                 "NBANNUITYDATA_QUALIFIED_VALUE_True": {0: 1, 1: 0},
#                 "NBAGENTDATA_ISPRIMARY_VALUE_False": {0: 0, 1: 0},
#                 "NBAGENTDATA_ISPRIMARY_VALUE_True": {0: 1, 1: 1},
#                 "NBREPLACEMENTINFODATA_ACCTYPE_VALUE_Annuity ": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_ACCTYPE_VALUE_Life Insurance": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_ACCTYPE_VALUE_Other": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_401(k)": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_403(b)": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_457 Deferred Comp.": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Beneficiary IRA": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Corporate Pension": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Defined Benefit": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Group TSA": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_IRA-Rollover": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_IRA-Simple": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Inheritance IRA (added by ERA)": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Inherited Non-Qualified": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Non-Qualified": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Not specified": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Pension Trust": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Roth IRA": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_SEP IRA": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Traditional IRA": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Trust": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_IRA-Spousal": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_401(a)": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_401(a) - Sched A": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Roth 401(k)": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Converted Roth IRA": {
#                     0: 0,
#                     1: 0,
#                 },
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Life Insurance": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Roth 403(b)": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_Custodial IRA": {0: 0, 1: 0},
#                 "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE_VUL NQ": {0: 0, 1: 0},
#                 "NBANNUITYDATA_EXPECTEDSURRPENALTY_VALUE_No": {0: 1, 1: 1},
#                 "NBANNUITYDATA_EXPECTEDSURRPENALTY_VALUE_Yes": {0: 0, 1: 0},
#                 "CASEHDRDATA_ISGROUP_VALUE_False": {0: 0, 1: 0},
#                 "CASEHDRDATA_ISGROUP_VALUE_True": {0: 1, 1: 1},
#                 "CASEHDRDATA_STATUS_VALUE_Completed": {0: 0, 1: 0},
#                 "CASEHDRDATA_STATUS_VALUE_Declined": {0: 0, 1: 0},
#                 "CASEHDRDATA_STATUS_VALUE_Issued": {0: 1, 1: 0},
#                 "CASEHDRDATA_STATUS_VALUE_New": {0: 0, 1: 1},
#                 "CASEHDRDATA_STATUS_VALUE_Pending Re-Issue": {0: 0, 1: 0},
#                 "CASEHDRDATA_STATUS_VALUE_Pending": {0: 0, 1: 0},
#                 "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE_Completed": {0: 0, 1: 0},
#                 "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE_Last Requirement": {0: 1, 1: 0},
#                 "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE_Requirement Review": {0: 0, 1: 1},
#                 "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE_Waiting/Pending": {0: 0, 1: 0},
#                 "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE_Blank": {0: 0, 1: 0},
#                 "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE_No": {0: 1, 1: 1},
#                 "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE_Yes": {0: 0, 1: 0},
#                 "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE_Blank": {0: 0, 1: 0},
#                 "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE_No": {0: 1, 1: 1},
#                 "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE_Yes": {0: 0, 1: 0},
#                 "CASEHDRDATA_ESIGNATUREIND_VALUE_False": {0: 0, 1: 0},
#                 "CASEHDRDATA_ESIGNATUREIND_VALUE_True": {0: 1, 1: 1},
#                 "FINANCIALGOALSDATA_ADEQUATEASSETS_VALUE_False": {0: 0, 1: 0},
#                 "FINANCIALGOALSDATA_ADEQUATEASSETS_VALUE_True": {0: 1, 1: 1},
#                 "APPCOVERAGESDATA_PRODMODELID_VALUE_RSL - Deferral": {0: 0, 1: 0},
#                 "class": {0: 1, 1: 0},
#             }
#         },
#     )
# )
