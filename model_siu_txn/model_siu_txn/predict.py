def predict(**kwargs):
    import pandas as pd
    import tempfile
    import boto3
    import pickle
    import json
    import warnings

    warnings.filterwarnings("ignore")

    cat_cols = [
        "CONTRACTDATA_WRITINGCODE",
        "TRXHDRDATA_TRXTYPEID_VALUE",
        "POLICYHDRDATA_PRODMODELID_VALUE",
        "EVENT",
        "POLICYHDRDATA_PLANOPTION",
        "POLICYHDRDATA_QUALIFIEDCODE_VALUE",
        "POLICYCOVHDRDATA_COVOPTION",
        "POLICYCOVHDRDATA_STATUS_VALUE",
        "NBANNUITYDATA_OWNERPARTYTYPE_VALUE",
        "TXN_TYPE",
    ]

    num_cols = [
        "uc2_employee_flag",
        "uc2_employee_total_policy_count",
        "uc7_employee_flag",
        "uc7_employee_zero_task_count",
        "uc7_employee_ratio",
        "uc8_employee_flag",
        "uc8_employee_policy_count",
        "uc8_employee_mean_change_days",
        "uc8_employee_ratio",
        "POLICYCOVHDRDATA_ISSUEAGE",
        "TRXRESDATA_FORMATACCTVALUE",
        "TRXRESDADATA_YTDWITHDRAWALS",
        "TRXRESDADATA_YTDSURRENDERCHARGE",
        "TRXRESDATA_AMTPROCESSED",
        "POLICYHDRDATA_FREELOOKPERIOD",
        "POLICYCOVHDRDATA_RETIREMENTAGE",
        "POLICYCOVHDRDATA_ISSUEDATEdiff",
        "POLICYHDRDATA_TIMESTAMPdiff",
        "CLAIMHDRDATA_DATEOFLOSSdiff",
        "CLAIMHDRDATA_DATEREPORTEDdiff",
        "CLAIMHDRDATA_TIMESTAMPdiff",
        "POLICYCOVHDRDATA_ENDDATEdiff",
        "POLICYCOVHDRDATA_TERMINATIONDATEdiff",
        "POLICYFUNDALLOCDATA_TIMESTAMPdiff",
        "NBANNUITYDATA_APPDATEISSUEDdiff",
        "CASEHDRDATA_EAPPDATARECEIVEDdiff",
        "POLICYCOVHDRDATA_duration",
        "no_employees",
        "predicted_disposition_inconsistent",
        "TXN_TYPE_non-financial-transaction",
        "webservices",
    ]

    bool_cols = [
        "POLICYHDRDATA_ISGROUP_VALUE",
        "POLICYHDRDATA_QUALIFIED_VALUE",
        "POLICYHDRDATA_REPLACEMENTFLAG_VALUE",
        "POLICYCOVHDRDATA_ISPARTOFCLAIM_VALUE",
        "emp_level_1",
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
    ]

    THRESHOLD = 0.3193858391936

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
        df["emp_level_1"] = df[emp_col].apply(lambda x: 1 if "470kjr" in x else 0)
        df["emp_level_2"] = df[emp_col].apply(lambda x: 1 if "470mkx" in x else 0)
        df["emp_level_3"] = df[emp_col].apply(lambda x: 1 if "470mex" in x else 0)
        df["emp_level_4"] = df[emp_col].apply(lambda x: 1 if "470whx" in x else 0)
        df["emp_level_5"] = df[emp_col].apply(lambda x: 1 if "470jax" in x else 0)
        df["emp_level_6"] = df[emp_col].apply(lambda x: 1 if "470srl" in x else 0)
        df["emp_level_7"] = df[emp_col].apply(lambda x: 1 if "725dpl" in x else 0)
        df["emp_level_8"] = df[emp_col].apply(lambda x: 1 if "470gtc" in x else 0)
        df["emp_level_9"] = df[emp_col].apply(lambda x: 1 if "303dwk" in x else 0)
        df["emp_level_10"] = df[emp_col].apply(lambda x: 1 if "470qda" in x else 0)
        df["emp_level_11"] = df[emp_col].apply(lambda x: 1 if "470hcl" in x else 0)
        df["emp_level_12"] = df[emp_col].apply(lambda x: 1 if "163fax" in x else 0)
        df["emp_level_13"] = df[emp_col].apply(lambda x: 1 if "470fbd" in x else 0)
        df["emp_level_14"] = df[emp_col].apply(lambda x: 1 if "470bpx" in x else 0)
        df["emp_level_15"] = df[emp_col].apply(lambda x: 1 if "470tbf" in x else 0)
        df["emp_level_16"] = df[emp_col].apply(lambda x: 1 if "725rtr" in x else 0)
        df["emp_level_17"] = df[emp_col].apply(lambda x: 1 if "205tdx" in x else 0)
        df["emp_level_18"] = df[emp_col].apply(lambda x: 1 if "302wrw" in x else 0)
        df["emp_level_19"] = df[emp_col].apply(lambda x: 1 if "110btw" in x else 0)
        df["emp_level_20"] = df[emp_col].apply(lambda x: 1 if "470knx" in x else 0)

        df["no_employees"] = df[emp_col].apply(lambda x: x.count(",") + 1)

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

    rs, catboost_encoder, lgb_clf = download_model_from_s3(model_bucket, model_key)
    pd_df = pd.DataFrame(kwargs.get("inputs").get("claim"))

    # process_nan's
    pd_df["NBANNUITYDATA_OWNERPARTYTYPE_VALUE"] = pd_df[
        "NBANNUITYDATA_OWNERPARTYTYPE_VALUE"
    ].fillna("Unknown")
    pd_df["uc8_employee_mean_change_days"] = pd_df[
        "uc8_employee_mean_change_days"
    ].fillna(-1)
    pd_df["CONTRACTDATA_WRITINGCODE"] = pd_df["CONTRACTDATA_WRITINGCODE"].fillna(
        "['Unknown']"
    )
    pd_df["CONTRACTDATA_WRITINGCODE"] = pd_df["CONTRACTDATA_WRITINGCODE"].map(
        lambda x: eval(x)[0]
    )
    pd_df["uc7_employee_ratio"] = pd_df["uc7_employee_ratio"].fillna(-1)
    pd_df["uc8_employee_ratio"] = pd_df["uc8_employee_ratio"].fillna(-1)

    # entities
    TXN_NUM = pd_df.pop("TRXNUM")

    # preprocess
    pd_df = preprocess_employeeid(pd_df)
    pd_df[bool_cols] = pd_df[bool_cols].astype("bool")
    pd_df[cat_cols] = pd_df[cat_cols].astype("category")
    pd_df = pd_df[cat_cols + num_cols + bool_cols]

    # Robust scaler
    pd_df.loc[:, num_cols] = rs.transform(pd_df.loc[:, num_cols].to_numpy())

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
            "inputDataSource": f"RANDN1209:0",
            "entityId": 'RAND1209',
            "predictedResult": predicted_claim,
        }
    ]


print(
    predict(
        model_name="SIU_TXN_model_v5",
        artifact=[
            {
                "dataId": "55dcc659-d0c5-42aa-b9bf-a0325a2997b9",
                "dataName": "combined_artifacts",
                "dataType": "artifact",
                "dataValue": "s3://siutempbucket/tariq/combined_txn_siu_c_v5.sav",
                "dataValueType": "str",
            }
        ],
        inputs={
            "claim": {
                "TRXHDRDATA_ID": {
                    0: "003bc4e3-5b59-4cdb-b15e-31b46b235771",
                    1: "005da770-b17c-420f-9657-e5a19c88b6c5",
                    2: "008f10d5-516d-44a8-a943-8d4aed4d86a3",
                    3: "0099103a-7df7-44db-ba39-a9425438fb9f",
                    4: "00a8b3a3-763f-47c7-948a-ec671457d291",
                },
                "POLICYNUMBER": {
                    0: "KXP0004011",
                    1: "K5P0002749",
                    2: "ATSIP00090",
                    3: "KXP0004617",
                    4: "ATMIP01695",
                },
                "TRXNUM": {0: 2502486, 1: 2502944, 2: 2569945, 3: 2444456, 4: 2209547},
                "EMPLOYEE_ID": {
                    0: "['500901F7D9E57D2FE0531100F20A6987', '2E8E6BFD0DA74EBFB2E622405EBE7980']",
                    1: "['500901F7D9E57D2FE0531100F20A6987', '201']",
                    2: "['500901F7D9E57D2FE0531100F20A6987', '270']",
                    3: "['500901F7D9E57D2FE0531100F20A6987', 'ECBEA0AC30040D6B0EAF81DA41D4C35']",
                    4: "['500901F7D9E57D2FE0531100F20A6987', '234']",
                },
                "V_SYSTEM_LOGON_ID": {
                    0: "['WebServices', '470kjr']",
                    1: "['WebServices', '470mex']",
                    2: "['WebServices', '303dwk']",
                    3: "['WebServices', '725dpl']",
                    4: "['WebServices', '470jax']",
                },
                "uc2_employee_flag": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0},
                "uc2_employee_total_policy_count": {
                    0: 2733,
                    1: 3794,
                    2: 1200,
                    3: 9033,
                    4: 3423,
                },
                "uc2_employee_withdrawl_policy_count": {
                    0: 149,
                    1: 176,
                    2: 58,
                    3: 3268,
                    4: 26,
                },
                "uc2_employee_policy_value": {
                    0: 39820658.66,
                    1: 171249994.57,
                    2: 15173219.78,
                    3: 440954763.71,
                    4: 8779215.69,
                },
                "uc2_employee_policy_withdrawl": {
                    0: 5038847.2,
                    1: 16128658.76,
                    2: 10818994.8,
                    3: 73471557.92,
                    4: 1151313.55,
                },
                "uc2_employee_ratio": {
                    0: 0.33184855233853,
                    1: 0.1447368421052631,
                    2: 0.9830508474576272,
                    3: 0.716352476983779,
                    4: 0.3376623376623376,
                },
                "uc7_employee_flag": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
                "uc7_employee_policy_count": {
                    0: 2733,
                    1: 3794,
                    2: 1200,
                    3: 9033,
                    4: 3423,
                },
                "uc7_employee_task_count": {
                    0: 3039,
                    1: 3556,
                    2: 1269,
                    3: 20656,
                    4: 2712,
                },
                "uc7_employee_high_task_count": {
                    0: 206,
                    1: 213,
                    2: 84,
                    3: 1968,
                    4: 142,
                },
                "uc7_employee_zero_task_count": {
                    0: 1370,
                    1: 2013,
                    2: 582,
                    3: 1911,
                    4: 1950,
                },
                "uc7_employee_ratio": {
                    0: 0.1511371973587674,
                    1: 0.1195957327344188,
                    2: 0.1359223300970873,
                    3: 0.2763268744734625,
                    4: 0.0964019008825526,
                },
                "uc8_employee_flag": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                "uc8_employee_policy_count": {
                    0: 2059,
                    1: 1328,
                    2: 299,
                    3: 236,
                    4: 3312,
                },
                "uc8_employee_change_policy_count": {0: 8, 1: 5, 2: 1, 3: 1, 4: 13},
                "uc8_employee_mean_change_days": {
                    0: 9.125,
                    1: 6.8,
                    2: 11.0,
                    3: 10.0,
                    4: 7.538461538461538,
                },
                "uc8_employee_party_change_count": {0: 12, 1: 6, 2: 1, 3: 1, 4: 16},
                "uc8_employee_ratio": {
                    0: 0.0038853812530354,
                    1: 0.0037650602409638,
                    2: 0.0033444816053511,
                    3: 0.0042372881355932,
                    4: 0.0039251207729468,
                },
                "CONTRACTDATA_WRITINGCODE": {
                    0: "['IND9913907']",
                    1: "['BDB9919660']",
                    2: "['BDB9900836']",
                    3: "['IND0006894']",
                    4: "['IND9914438']",
                },
                "numerator": {0: 3, 1: 3, 2: 3, 3: 3, 4: 3},
                "denominator": {0: 3, 1: 3, 2: 3, 3: 3, 4: 3},
                "score": {
                    0: 0.6666666666666666,
                    1: 0.3333333333333333,
                    2: 0.6666666666666666,
                    3: 0.6666666666666666,
                    4: 0.3333333333333333,
                },
                "predicted_disposition": {
                    0: "consistent",
                    1: "consistent",
                    2: "consistent",
                    3: "consistent",
                    4: "consistent",
                },
                "eligible_hypothesis": {
                    0: "3/3",
                    1: "3/3",
                    2: "3/3",
                    3: "3/3",
                    4: "3/3",
                },
                "TRXHDRDATA_TRXSTATUS_VALUE": {
                    0: "Completed",
                    1: "Completed",
                    2: "Completed",
                    3: "Completed",
                    4: "Completed",
                },
                "TRXHDRDATA_PROCESSEDDATE": {
                    0: "2021-05-13",
                    1: "2021-05-16",
                    2: "2021-08-04",
                    3: "2021-03-08",
                    4: "2020-10-19",
                },
                "TRXHDRDATA_TRXTYPEID_VALUE": {
                    0: "Death Claim",
                    1: "Cost Basis Adjustment",
                    2: "Death Claim Disbursement",
                    3: "Partial Surrender",
                    4: "Death Claim Disbursement",
                },
                "POLICYHDRDATA_PRODMODELID_VALUE": {
                    0: "RSL - Keystone",
                    1: "RSL - Keystone",
                    2: "RSL - Eleos",
                    3: "RSL - Keystone",
                    4: "RSL - Eleos",
                },
                "POLICYCOVHDRDATA_ISSUEDATE": {
                    0: "2018-07-02",
                    1: "2018-08-01",
                    2: "2017-11-27",
                    3: "2018-10-01",
                    4: "2018-05-08",
                },
                "POLICYCOVHDRDATA_ISSUEAGE": {0: 74, 1: 61, 2: 85, 3: 71, 4: 84},
                "TRXRESDATA_FORMATACCTVALUE": {
                    0: 0.0,
                    1: 54490.78,
                    2: 0.0,
                    3: 11626.39,
                    4: 0.0,
                },
                "TRXRESDADATA_YTDWITHDRAWALS": {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 79500.0,
                    4: 0.0,
                },
                "TRXRESDADATA_YTDSURRENDERCHARGE": {
                    0: 0.0,
                    1: 0.0,
                    2: 0.0,
                    3: 5630.99,
                    4: 0.0,
                },
                "TRXRESDATA_AMTPROCESSED": {
                    0: 120303.56,
                    1: 0.0,
                    2: 177972.19,
                    3: 10500.0,
                    4: 24808.07,
                },
                "EVENT": {
                    0: "PolicyCreateNotes",
                    1: "PolicyHdrTaskInsert",
                    2: "PolicyUpdateDeathClaim",
                    3: "PolicyUpdate",
                    4: "PolicySnapshot",
                },
                "POLICYHDRDATA_FREELOOKPERIOD": {0: 30, 1: 30, 2: 20, 3: 30, 4: 20},
                "POLICYHDRDATA_ISGROUP_VALUE": {
                    0: False,
                    1: False,
                    2: False,
                    3: False,
                    4: False,
                },
                "POLICYHDRDATA_PLANOPTION": {
                    0: "SPDA_10Yr",
                    1: "SPDA_5Yr",
                    2: "SPDA",
                    3: "SPDA_10Yr",
                    4: "SPDA",
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE": {
                    0: "Non-Qualified",
                    1: "Non-Qualified",
                    2: "Non-Qualified",
                    3: "Non-Qualified",
                    4: "Non-Qualified",
                },
                "POLICYHDRDATA_QUALIFIED_VALUE": {
                    0: False,
                    1: False,
                    2: False,
                    3: False,
                    4: False,
                },
                "POLICYHDRDATA_REPLACEMENTFLAG_VALUE": {
                    0: True,
                    1: True,
                    2: False,
                    3: True,
                    4: False,
                },
                "POLICYHDRDATA_APPLICATIONDATE": {
                    0: "2018-06-12",
                    1: "2018-06-22",
                    2: "2017-11-22",
                    3: "2018-08-16",
                    4: "2018-05-03",
                },
                "POLICYHDRDATA_TIMESTAMP": {
                    0: "2021-05-13 19:45:01.413850300",
                    1: "2019-08-01 23:03:37.404624200",
                    2: "2021-05-26 20:13:33.321574900",
                    3: "2021-08-18 20:10:30.309824800",
                    4: "2020-10-30 21:09:03.360134400",
                },
                "CLAIMHDRDATA_BENEFITAMT": {
                    0: 120303.56,
                    1: None,
                    2: 177972.19,
                    3: None,
                    4: 72964.92,
                },
                "CLAIMHDRDATA_DATEOFLOSS": {
                    0: None,
                    1: None,
                    2: None,
                    3: None,
                    4: None,
                },
                "CLAIMHDRDATA_DATEREPORTED": {
                    0: "2021-03-08",
                    1: None,
                    2: "2021-05-26",
                    3: None,
                    4: "2020-10-05",
                },
                "CLAIMHDRDATA_SPOUSALCONTINUATION_VALUE": {
                    0: False,
                    1: None,
                    2: False,
                    3: None,
                    4: False,
                },
                "CLAIMHDRDATA_STATUS_VALUE": {
                    0: "Incurred",
                    1: None,
                    2: "Incurred",
                    3: None,
                    4: "Incurred",
                },
                "CLAIMHDRDATA_TIMESTAMP": {
                    0: "2021-05-13 14:24:24.891471200",
                    1: None,
                    2: "2021-08-04 11:17:55.728954400",
                    3: None,
                    4: "2020-10-19 11:07:28.089739000",
                },
                "CLAIMHDRDATA_TOTALBASIS": {
                    0: 85902.59,
                    1: None,
                    2: 164319.74,
                    3: None,
                    4: 70191.87,
                },
                "POLICYCOVHDRDATA_COVOPTION": {
                    0: "BaseCov_SPEIDA",
                    1: "BaseCov_SPEIDA",
                    2: "BaseCov_SPDA",
                    3: "BaseCov_SPEIDA",
                    4: "BaseCov_SPDAMVA",
                },
                "POLICYCOVHDRDATA_EFFECTIVEDATE": {
                    0: "2018-07-02",
                    1: "2018-08-01",
                    2: "2017-11-27",
                    3: "2018-10-01",
                    4: "2018-05-08",
                },
                "POLICYCOVHDRDATA_STARTDATE": {
                    0: "2018-06-26",
                    1: "2018-07-26",
                    2: "2017-11-27",
                    3: "2018-09-20",
                    4: "2018-05-08",
                },
                "POLICYCOVHDRDATA_ENDDATE": {
                    0: None,
                    1: None,
                    2: None,
                    3: None,
                    4: None,
                },
                "POLICYCOVHDRDATA_ISPARTOFCLAIM_VALUE": {
                    0: True,
                    1: False,
                    2: True,
                    3: False,
                    4: True,
                },
                "POLICYCOVHDRDATA_RETIREMENTAGE": {0: 70, 1: 70, 2: 70, 3: 70, 4: 70},
                "POLICYCOVHDRDATA_STATUS_VALUE": {
                    0: "Active",
                    1: "Active",
                    2: "Active",
                    3: "Active",
                    4: "Active",
                },
                "POLICYCOVHDRDATA_TERMINATIONDATE": {
                    0: "2029-07-02",
                    1: "2042-08-01",
                    2: "2027-11-27",
                    3: "2032-10-01",
                    4: "2028-05-08",
                },
                "POLICYCOVHDRDATA_TIMESTAMP": {
                    0: "2021-05-13 19:44:58.210717900",
                    1: "2022-08-01 21:25:41.531961900",
                    2: "2021-05-26 20:13:35.040326800",
                    3: "2021-07-13 09:36:26.062484600",
                    4: "2020-10-30 21:08:58.485107300",
                },
                "POLICYFUNDALLOCDATA_EFFECTIVEDATE": {
                    0: "2018-07-02",
                    1: "2018-08-01",
                    2: "2017-11-27",
                    3: "2018-10-01",
                    4: "2018-05-08",
                },
                "POLICYFUNDALLOCDATA_TIMESTAMP": {
                    0: "2018-06-27 11:08:20.665843600",
                    1: "2018-07-31 17:42:12.316134500",
                    2: "2017-11-29 09:42:35.212082000",
                    3: "2018-09-21 14:23:53.776303500",
                    4: "2018-05-11 13:05:34.725706000",
                },
                "POLICYFUNDALLOCFUNDALLOCATIONHDRDATA_TIMESTAMP": {
                    0: "2018-06-27 11:08:09.000000",
                    1: "2018-07-31 17:42:07.000000",
                    2: "2017-11-29 09:42:30.000000",
                    3: "2018-09-21 14:23:44.000000",
                    4: "2018-05-11 13:05:30.000000",
                },
                "NBANNUITYDATA_OWNERPARTYTYPE_VALUE": {
                    0: "Person",
                    1: "Person",
                    2: "Person",
                    3: "Person",
                    4: "Person",
                },
                "NBANNUITYDATA_APPDATEISSUED": {
                    0: "2018-06-14",
                    1: "2018-07-02",
                    2: "2017-11-27",
                    3: "2018-08-23",
                    4: "2018-05-08",
                },
                "TXN_TYPE": {
                    0: "financial-transaction",
                    1: "financial-transaction",
                    2: "financial-transaction",
                    3: "financial-transaction",
                    4: "financial-transaction",
                },
                "CASEHDRDATA_APPSIGNDATE": {
                    0: "2018-06-12",
                    1: "2018-06-22",
                    2: "2017-11-22",
                    3: "2018-08-16",
                    4: "2018-05-03",
                },
                "CASEHDRDATA_DATEISSUED": {
                    0: "2018-07-02",
                    1: "2018-08-01",
                    2: "2017-11-27",
                    3: "2018-10-01",
                    4: "2018-05-08",
                },
                "CASEHDRDATA_DATERECEIVED": {
                    0: "2018-06-14",
                    1: "2018-07-02",
                    2: "2017-11-27",
                    3: "2018-08-23",
                    4: "2018-05-08",
                },
                "CASEHDRDATA_EAPPDATARECEIVED": {
                    0: None,
                    1: None,
                    2: None,
                    3: None,
                    4: None,
                },
                "CASEHDRDATA_RATELOCKEFFECTIVEDATE": {
                    0: "2018-07-02",
                    1: "2018-08-01",
                    2: "2017-11-27",
                    3: "2018-10-01",
                    4: "2018-05-08",
                },
                "CASEHDRDATA_TIMESTAMP": {
                    0: "2018-06-27 11:08:20.415837300",
                    1: "2018-07-31 17:42:12.206745400",
                    2: "2017-11-29 09:42:35.137825000",
                    3: "2018-09-21 14:23:53.541945900",
                    4: "2018-05-11 13:05:34.553822200",
                },
                "FINANCIALGOALSDATA_TIMESTAMP": {
                    0: "2018-06-14 13:07:01.747150100",
                    1: "2018-07-03 14:03:36.032137700",
                    2: "2017-11-27 12:07:47.464163000",
                    3: "2018-08-23 14:14:05.591085300",
                    4: "2018-05-08 12:05:58.149834500",
                },
                "POLICYCOVHDRDATA_ISSUEDATEdiff": {
                    0: 1046,
                    1: 1019,
                    2: 1346,
                    3: 889,
                    4: 895,
                },
                "POLICYHDRDATA_APPLICATIONDATEdiff": {
                    0: 1066,
                    1: 1059,
                    2: 1351,
                    3: 935,
                    4: 900,
                },
                "POLICYHDRDATA_TIMESTAMPdiff": {0: 0, 1: 654, 2: 70, 3: -1, 4: -1},
                "CLAIMHDRDATA_DATEOFLOSSdiff": {0: -1, 1: -1, 2: -1, 3: -1, 4: -1},
                "CLAIMHDRDATA_DATEREPORTEDdiff": {0: 66, 1: -1, 2: 70, 3: -1, 4: 14},
                "CLAIMHDRDATA_TIMESTAMPdiff": {0: 0, 1: -1, 2: 0, 3: -1, 4: 0},
                "POLICYCOVHDRDATA_EFFECTIVEDATEdiff": {
                    0: 1046,
                    1: 1019,
                    2: 1346,
                    3: 889,
                    4: 895,
                },
                "POLICYCOVHDRDATA_STARTDATEdiff": {
                    0: 1052,
                    1: 1025,
                    2: 1346,
                    3: 900,
                    4: 895,
                },
                "POLICYCOVHDRDATA_ENDDATEdiff": {0: -1, 1: -1, 2: -1, 3: -1, 4: -1},
                "POLICYCOVHDRDATA_TERMINATIONDATEdiff": {
                    0: -1,
                    1: -1,
                    2: -1,
                    3: -1,
                    4: -1,
                },
                "POLICYCOVHDRDATA_TIMESTAMPdiff": {0: 0, 1: -1, 2: 70, 3: -1, 4: -1},
                "POLICYFUNDALLOCDATA_EFFECTIVEDATEdiff": {
                    0: 1046,
                    1: 1019,
                    2: 1346,
                    3: 889,
                    4: 895,
                },
                "POLICYFUNDALLOCDATA_TIMESTAMPdiff": {
                    0: 1051,
                    1: 1020,
                    2: 1344,
                    3: 899,
                    4: 892,
                },
                "POLICYFUNDALLOCFUNDALLOCATIONHDRDATA_TIMESTAMPdiff": {
                    0: 1051,
                    1: 1020,
                    2: 1344,
                    3: 899,
                    4: 892,
                },
                "NBANNUITYDATA_APPDATEISSUEDdiff": {
                    0: 1064,
                    1: 1049,
                    2: 1346,
                    3: 928,
                    4: 895,
                },
                "CASEHDRDATA_APPSIGNDATEdiff": {
                    0: 1066,
                    1: 1059,
                    2: 1351,
                    3: 935,
                    4: 900,
                },
                "CASEHDRDATA_DATEISSUEDdiff": {
                    0: 1046,
                    1: 1019,
                    2: 1346,
                    3: 889,
                    4: 895,
                },
                "CASEHDRDATA_DATERECEIVEDdiff": {
                    0: 1064,
                    1: 1049,
                    2: 1346,
                    3: 928,
                    4: 895,
                },
                "CASEHDRDATA_EAPPDATARECEIVEDdiff": {0: -1, 1: -1, 2: -1, 3: -1, 4: -1},
                "CASEHDRDATA_RATELOCKEFFECTIVEDATEdiff": {
                    0: 1046,
                    1: 1019,
                    2: 1346,
                    3: 889,
                    4: 895,
                },
                "CASEHDRDATA_TIMESTAMPdiff": {
                    0: 1051,
                    1: 1020,
                    2: 1344,
                    3: 899,
                    4: 892,
                },
                "FINANCIALGOALSDATA_TIMESTAMPdiff": {
                    0: 1064,
                    1: 1048,
                    2: 1346,
                    3: 928,
                    4: 895,
                },
                "POLICYCOVHDRDATA_duration": {0: -1, 1: -1, 2: -1, 3: -1, 4: -1},
                "no_employees": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                "predicted_disposition_consistent": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "predicted_disposition_inconsistent": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                "TRXHDRDATA_TRXSTATUS_VALUE_Completed": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                "TRXHDRDATA_TRXTYPEID_VALUE_Cost Basis Adjustment": {
                    0: 0,
                    1: 1,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim": {
                    0: 1,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim - 5YD": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim - LD": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim Disbursement": {
                    0: 0,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 1,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Free Look Cancel": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Full Surrender": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Issue": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "TRXHDRDATA_TRXTYPEID_VALUE_Partial Surrender": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Spousal Continuation": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Value Adjustment": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim - 10YD": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Annuitization": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Dividend Release": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Annuity Payment": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Death Claim Annuitization": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_DOB Correction": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Apollo": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Apollo 93 BV": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Apollo 93 MVA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Argus BV": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Argus MVA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Cornerstone": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Deferral": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Elektra 5 6 7": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Eleos": {
                    0: 0,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 1,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Eleos 96 BV": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Eleos 96 MVA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - FPDA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - John Alden": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Keystone": {
                    0: 1,
                    1: 1,
                    2: 0,
                    3: 1,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Reliance Guarantee": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_SSL - Converted": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Reliance Accumulator": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_PRODMODELID_VALUE_RSL - Deferral Annuitization": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "EVENT_ClaimPayoutInsert": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_ClaimPayoutUpdate": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_Policy-StatusChange": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyChangeAgentRecord": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyChangeAgentRecordUpdate": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyCovHdrPartyRoleInsert": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyCovHdrPartyRoleUpdate": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyCreateDeathClaim": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyCreateNotes": {0: 1, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyCreatePayoutDocumentation": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyDeletePayoutDocumentation": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyHdrTaskInsert": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyInsertFundsAllocation": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyPartyRolesUpdate": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicySnapshot": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
                "EVENT_PolicyUpdate": {0: 0, 1: 0, 2: 0, 3: 1, 4: 0},
                "EVENT_PolicyUpdateDeathClaim": {0: 0, 1: 0, 2: 1, 3: 0, 4: 0},
                "EVENT_PolicyUpdateMaturityDate": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyUpdateRoles": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_ClaimHdrUserNotesInsert": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyHdrCashDtlInsert": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyUpdateFundsAllocation": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_PolicyUpdatePayoutDocumentation": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "EVENT_CreatePolicy": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_ISGROUP_VALUE_False": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                "POLICYHDRDATA_ISGROUP_VALUE_True": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_PLANOPTION_FPDA": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_PLANOPTION_SPDA": {0: 0, 1: 0, 2: 1, 3: 0, 4: 1},
                "POLICYHDRDATA_PLANOPTION_SPDA_10Yr": {0: 1, 1: 0, 2: 0, 3: 1, 4: 0},
                "POLICYHDRDATA_PLANOPTION_SPDA_10yr": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_PLANOPTION_SPDA_5Yr": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_PLANOPTION_SPDA_5yr": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_PLANOPTION_SPDA_6yr": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_PLANOPTION_SPDA_7Yr": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_PLANOPTION_SPDA_7yr": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Beneficiary IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Beneficiary IRA 5YD": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Beneficiary IRA LD": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Custodial IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Inherited Non-Qualified": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Non-Qualified": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Pension Trust": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Roth IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_SEP IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Traditional IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Beneficiary IRA 10YD": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Inherited Roth IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Custodial Roth IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Not specified": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIEDCODE_VALUE_Converted Roth IRA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "POLICYHDRDATA_QUALIFIED_VALUE_False": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                "POLICYHDRDATA_QUALIFIED_VALUE_True": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYHDRDATA_REPLACEMENTFLAG_VALUE_False": {
                    0: 0,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 1,
                },
                "POLICYHDRDATA_REPLACEMENTFLAG_VALUE_True": {
                    0: 1,
                    1: 1,
                    2: 0,
                    3: 1,
                    4: 0,
                },
                "CLAIMHDRDATA_SPOUSALCONTINUATION_VALUE_False": {
                    0: 1,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 1,
                },
                "CLAIMHDRDATA_SPOUSALCONTINUATION_VALUE_True": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "CLAIMHDRDATA_STATUS_VALUE_Cancelled": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "CLAIMHDRDATA_STATUS_VALUE_Incurred": {0: 1, 1: 0, 2: 1, 3: 0, 4: 1},
                "CLAIMHDRDATA_STATUS_VALUE_Open": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "CLAIMHDRDATA_STATUS_VALUE_Rejected": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYCOVHDRDATA_COVOPTION_BaseCov_SPDA": {
                    0: 0,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 0,
                },
                "POLICYCOVHDRDATA_COVOPTION_BaseCov_SPDAMVA": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 1,
                },
                "POLICYCOVHDRDATA_COVOPTION_BaseCov_SPEIDA": {
                    0: 1,
                    1: 1,
                    2: 0,
                    3: 1,
                    4: 0,
                },
                "POLICYCOVHDRDATA_ISPARTOFCLAIM_VALUE_False": {
                    0: 0,
                    1: 1,
                    2: 0,
                    3: 1,
                    4: 0,
                },
                "POLICYCOVHDRDATA_ISPARTOFCLAIM_VALUE_True": {
                    0: 1,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 1,
                },
                "POLICYCOVHDRDATA_STATUS_VALUE_Active": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                "POLICYCOVHDRDATA_STATUS_VALUE_Pending": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "POLICYCOVHDRDATA_STATUS_VALUE_Issued": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "NBANNUITYDATA_OWNERPARTYTYPE_VALUE_Organization": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "NBANNUITYDATA_OWNERPARTYTYPE_VALUE_Person": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                },
                "TXN_TYPE_financial-transaction": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                "TXN_TYPE_non-financial-transaction": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "TRXHDRDATA_TRXTYPEID_VALUE_Systematic Withdrawal": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TXN_TYPE_NON-FINANCIAL": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "TRXHDRDATA_TRXTYPEID_VALUE_Anniversary": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Calendar Year End": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Pending Death Claim": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "TRXHDRDATA_TRXTYPEID_VALUE_Premium": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                "TRXHDRDATA_TRXTYPEID_VALUE_Renewal Transaction": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                },
                "feedback_field": {
                    0: "modelType",
                    1: "modelType",
                    2: "modelType",
                    3: "other",
                    4: "modelType",
                },
                "feedback_reason": {
                    0: "Correct Date of Death and date the death certificate was received input",
                    1: "Cost basis input based on direct transfer indication listed on incoming transfer paperwork, considered new money. ",
                    2: "Claim was processed correctly per beneficiary request",
                    3: "Policyowner requested to have funds sent via EFT and a check was mailed",
                    4: "Claim was processed correctly per beneficiary request",
                },
                "class": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            },
        },
    )
)
