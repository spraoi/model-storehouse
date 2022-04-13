import pandas as pd
import dateutil.parser

role = ""
channel = ""
target_column = ""
eligible_column = ""
predicted_column = ""
deleted_cols = []

categorical_columns = [
    "modelType",
    "productType",
    "ownerType",
    "template",
    "NBREPLACEMENTINFODATA_ACCTYPE_VALUE",
    "NBREPLACEMENTINFODATA_EXCHANGETYPE_VALUE",
    "NBPAYMENTSDATA_METHOD",
    "NBANNUITYDATA_APPSTATUS_VALUE",
    "NBANNUITYDATA_APPTYPE_VALUE",
    "NBANNUITYDATA_ISREPLACEMENT_VALUE",
    "NBANNUITYDATA_QUALIFIED_VALUE",
    "NBAGENTDATA_ISPRIMARY_VALUE",
    "NBANNUITYDATA_SYSTEMATICWITHDRAWAL_VALUE",
    "NBANNUITYDATA_ROLLOVER_VALUE",
    "NBANNUITYDATA_PURPOSEOFACCOUNT_VALUE",
    "NBANNUITYDATA_PAYMENTMETHOD_VALUE",
    "NBANNUITYDATA_OWNERRELATION_VALUE",
    "NBANNUITYDATA_OWNERORGTYPE_VALUE",
    "NBANNUITYDATA_OWNERANNUITANTFLAG_VALUE",
    "NBANNUITYDATA_ISGROUP_VALUE",
    "NBANNUITYDATA_EXPECTEDSURRPENALTY_VALUE",
    "NBREPLACEMENTINFODATA_PARTIALORFULL_VALUE",
    "NBREPLACEMENTINFODATA_PAYMENTSOURCETYPE_CODE",
    "NBANNUITYDATA_CHECKFORMS_VALUE",
    "NBANNUITYDATA_COMMOPTIONTYPE_VALUE",
    "NBANNUITYDATA_ANNUITYPOLICYTYPE",
    "NBANNUITYDATA_APPSIGNSTATE_VALUE",
    "NBANNUITYDATA_ANNUITYPOLICYOPTION",
    "NBANNUITYDATA_ANNUITANTMARSTAT_VALUE",
    "NBANNUITYDATA_ADEQUATEASSETS_VALUE",
    "NBANNUITYDATA_ACCOUNTTYPE_VALUE",
    "NBPAYMENTSDATA_PREMIUMTYPE_CODE",
    "NBREPLACEMENTINFODATA_PREVPLANTYPE_VALUE",
    "NBREPLACEMENTINFODATA_REPLCO1035_VALUE",
    "NBANNUITYDATA_ESIGNATUREIND_VALUE",
    "CASEHDRDATA_ISGROUP_VALUE",
    "CASEHDRDATA_STATUS_VALUE",
    "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE",
    "FINANCIALGOALSDATA_ADEQUATEASSETS_VALUE",
    "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE",
    "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE",
    "CASEHDRDATA_ESIGNATUREIND_VALUE",
]
date_columns = [
    "receivedDate",
    "issueDate",
    "NBANNUITYNBAGENTDATA_TIMESTAMP",
    "NBANNUITYNBREPLACEMENTINFODATA_TIMESTAMP",
    "NBREPLACEMENTINFODATA_TIMESTAMP",
    "NBREPLACEMENTINFODATA_PAPERWORKRECEIVEDDATE",
    "FORMMANAGERDATA_TIMESTAMP",
    "NBANNUITYNBHDRDATA_TIMESTAMP",
    "NBANNUITYFORMMANAGERDATA_TIMESTAMP",
    "NBPAYMENTSDATA_EFFECTIVEDATE",
    "NBPAYMENTSDATA_TIMESTAMP",
    "NBANNUITYDATA_ANNUITYDATE",
    "NBANNUITYDATA_APPDATEISSUED",
    "NBANNUITYDATA_APPDATERECEIVED",
    "NBANNUITYDATA_APPSIGNDATE",
    "NBANNUITYDATA_CHECKENTRYDATE",
    "NBANNUITYDATA_OWNERTRUSTDATE",
    "NBANNUITYDATA_ROTHIRAINCEPTIONDATE",
    "NBANNUITYDATA_TIMESTAMP",
    "NBANNUITYNBPAYMENTSDATA_TIMESTAMP",
    "NBAGENTDATA_TIMESTAMP",
    "NBDECEDENTDATA_DECEDENTDATEOFDEATH",
    "NBANNUITYNBFUNDALLOCHDRDATA_TIMESTAMP",
    "DATE_LOADED",
    "CASEHDRDATA_APPSIGNDATE",
    "CASEHDRDATA_DATEISSUED",
    "CASEHDRDATA_DATERECEIVED",
    "CASEHDRDATA_RATELOCKEFFECTIVEDATE",
    "CASEHDRDATA_TIMESTAMP",
    "FINANCIALGOALSDATA_TIMESTAMP",
    "CASEHDRDATA_EAPPDATARECEIVED",
]
num_columns = [
    "policyValue",
    "policyAge",
    "policyWithdrawals",
    "surrenderCharges",
    "policyAge",
    "suitabilityScore",
    "NBREPLACEMENTINFODATA_ESTIMATEDVALUE",
    "NBPAYMENTSDATA_AMOUNT",
    "NBPAYMENTSDATA_COMMISSIONRETAINED",
    "NBPAYMENTSDATA_NETCOMMISSIONS",
    "NBANNUITYDATA_EXPECTEDPREM",
    "NBANNUITYDATA_COMMISSIONSWITHHELD",
    "NBAGENTDATA_PERCENTAGE",
    "NBANNUITYDATA_ANNUITANTAGE",
    "NBANNUITYDATA_CASHWITHAPP",
    "NBANNUITYDATA_CHARGESINCURREPLACE",
    "FINANCIALGOALSDATA_SUITABILITYSCORE",
    "FINANCIALGOALSDATA_FORMATNETWORTH",
    "FINANCIALGOALSDATA_EXPECTEDPREM",
    "FINANCIALGOALSDATA_CHARGESINCURREPLACE",
]
dist_columns = ["agentIds", "employeeIds"]
id_column = "shortId"

prediction_map = {"consistent": 0, "inconsistent": 1}

siu_feat_diff = [
    "CASEHDRDATA_ISGROUP_VALUE",
    "template",
    "FINANCIALGOALSDATA_REVERSEMORTGAGE_CODE",
    "NBANNUITYDATA_CHECKFORMS_VALUE",
    "FINANCIALGOALSDATA_ADEQUATEASSETS_VALUE",
    "CASEHDRDATA_ESIGNATUREIND_VALUE",
    "modelType",
    "CASEHDRDATA_UNDERWRITINGSTATUS_VALUE",
    "NBANNUITYDATA_APPTYPE_VALUE",
    "FINANCIALGOALSDATA_EXPECTEDSURRPENALTY_CODE",
    "CASEHDRDATA_STATUS_VALUE",
    "NBREPLACEMENTINFODATA_EXCHANGETYPE_VALUE",
]

ops_feat_diff = [
    "NBANNUITYDATA_ESIGNATUREIND_VALUE",
    "NBANNUITYDATA_OWNERANNUITANTFLAG_VALUE",
    "ownerType",
    "template",
]

date_diff = [
    "receivedDate",
    "issueDate",
    "CASEHDRDATA_APPSIGNDATE",
    "CASEHDRDATA_DATEISSUED",
    "CASEHDRDATA_DATERECEIVED",
    "CASEHDRDATA_RATELOCKEFFECTIVEDATE",
    "CASEHDRDATA_TIMESTAMP",
    "FINANCIALGOALSDATA_TIMESTAMP",
    "CASEHDRDATA_EAPPDATARECEIVED",
]


def preprocess_NumColumns(df, num_columns=num_columns):
    for col in num_columns:
        df[col] = df[col].astype(str).replace(",", "", regex=True)
        df[col] = df[col].apply(pd.to_numeric, errors="coerce")
    return df


def preprocess_CatColumns(df, categorical_columns):
    data_dummy_df = pd.get_dummies(df, columns=categorical_columns)
    return data_dummy_df


def preprocess_DateColumns(df, date_columns=date_columns):
    date_columns = [
        "receivedDate",
        "issueDate",
        "CASEHDRDATA_APPSIGNDATE",
        "CASEHDRDATA_DATEISSUED",
        "CASEHDRDATA_DATERECEIVED",
        "CASEHDRDATA_RATELOCKEFFECTIVEDATE",
        "CASEHDRDATA_TIMESTAMP",
        "FINANCIALGOALSDATA_TIMESTAMP",
        "CASEHDRDATA_EAPPDATARECEIVED",
    ]
    for date_cols in date_columns:
        df[date_cols] = df[date_cols].replace("NaN",np.nan)
        df[date_cols] = (
            df[date_cols].fillna("1900-01-01").apply(lambda x: dateutil.parser.parse(x))
        )

    for cols in date_columns:
        diff_cols = cols + "diff"
        df[diff_cols] = df[date_cols] - df["issueDate"]
        df[diff_cols] = df[diff_cols].dt.days.apply(lambda x: -1 if x > 30000 else x)

    df.drop(columns=date_columns, inplace=True)
    return df


def preprocess_agentID(df, agent_col="agentIds"):
    try:
        df["no_agents"] = df[agent_col].apply(lambda x: x.count(",") + 1)
    except AttributeError:
        df["no_agents"] = 0
    df[["no_agents"]] = df[["no_agents"]].apply(pd.to_numeric, errors="coerce")
    df = df.drop(agent_col, axis=1)
    return df


def preprocess_employeeID(df, emp_col="employeeIds"):
    df["employeeIds"] = df["employeeIds"].fillna("None")
    df["webservices"] = df[emp_col].apply(lambda x: 1 if "WebServices" in x else 0)
    df["no_employees"] = df[emp_col].apply(lambda x: x.count(",") + 1)
    df[["webservices", "no_employees"]] = df[["webservices", "no_employees"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.drop(emp_col, axis=1)
    return df


def preprocess_hypothesis(df, role=role):
    eligible_col = role + ".eligibleHypothesis"
    if eligible_col not in deleted_cols:
        try:
            df["eligible_num"] = df[eligible_col].apply(lambda x: x.split("/")[0])
            df["eligible_denom"] = df[eligible_col].apply(lambda x: x.split("/")[1])
        except KeyError:
            df["eligible_num"] = -1
            df["eligible_denom"] = -1
        df[["eligible_num", "eligible_denom"]] = df[
            ["eligible_num", "eligible_denom"]
        ].apply(pd.to_numeric, errors="coerce")
        df = df.drop(eligible_col, axis=1)
    return df


def preprocess_ruleprediction(df, role=role):
    predicted_column = role + ".predictedDisposition"
    df[predicted_column] = df[predicted_column].replace(
        {"consistent": 0, "inconsistent": 1}
    )
    return df


def preprocess(df, categorical_columns, role=role):

    df = preprocess_NumColumns(df)
    df = preprocess_DateColumns(df)

    df = df.loc[:, ~df.columns.duplicated()]

    if len(categorical_columns):
        df = preprocess_CatColumns(df, categorical_columns=categorical_columns)

    if "agentIds" not in deleted_cols:
        df = preprocess_agentID(df)
    if "employeeIds" not in deleted_cols:
        df = preprocess_employeeID(df)
    if eligible_column not in deleted_cols:
        df = preprocess_hypothesis(df, role)
    df = preprocess_ruleprediction(df, role)

    # target_column = role+'.actualDisposition'
    # df.rename(columns={target_column:'class'},inplace=True)
    df.fillna(-1, inplace=True)
    return df


def Diff(li1, li2):

    return list(set(li1) - set(li2))
