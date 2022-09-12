import pandas as pd
import numpy as np
import boto3
import joblib
import tempfile
import onnxruntime
# from tensorflow.keras.models import load_model


def resolve_formatting(df, date_cols, numeric_cols):
    """
    function to ensure correct date and numeric formatting
    """
    for col in list(df.columns):
        if col in date_cols:
            df.loc[:, col] = pd.to_datetime(df.loc[:, col], errors="coerce")
        elif col in numeric_cols:
            df.loc[:, col] = pd.to_numeric(df.loc[:, col])

    return df


def get_date_diff(col1, col2, interval):
    """difference between dates specified by the interval in ['Y','D','M']
    col1 and col2 are date colummns and col2 > col1"""

    return (col2 - col1) / np.timedelta64(1, interval)


def add_policy_tenure_to_df(df):
    """
    returns a df with policy tenure column appended
    """
    df["policy_tenure"] = get_date_diff(
        df["Policy Effective Date"], df["Policy Termination Date"], interval="D"
    )
    df["policy_tenure_2"] = get_date_diff(
        df["Policy Effective Date"], df["Received Date"], interval="D"
    )
    df.loc[df["policy_tenure"].isnull(), "policy_tenure"] = df.loc[
        df["policy_tenure"].isnull(), "policy_tenure_2"
    ]
    df.drop("policy_tenure_2", axis=1, inplace=True)
    return df


def add_prognosis_days_to_df(df):
    """
    returns a df with prognosis days column appeneded
    """

    df["prognosis_days"] = get_date_diff(
        df["Loss Date"], df["Duration Date"], interval="D"
    )
    col = "prognosis_days"

    df.loc[(df[col].isnull()) | (df[col] <= 0), col] = (
        pd.to_numeric(df.loc[(df[col].isnull()) | (df[col] <= 0), "Duration Months"])
        * 30
    )
    df.loc[df[col] <= 0, col] = np.nan
    return df


def tokenize_pd_code(df):

    pd_tokens = list(
        df["Primary Diagnosis Code"].fillna("_na").apply(lambda x: x.split("."))
    )
    pd_tokens_df = pd.DataFrame(pd_tokens, columns=["pd_code_1", "pd_code_2"])
    df.reset_index(inplace=True)
    df.loc[:, "pd_code_1"] = pd_tokens_df["pd_code_1"].copy()
    df.loc[:, "pd_code_2"] = pd_tokens_df["pd_code_2"].copy()
    return df


def pre_process_data(df):

    ID = ["Claim Identifier"]
    DATES = [
        "Received Date",
        "Loss Date",
        "Insured Hire Date",
        "Duration Date",
        "Policy Effective Date",
        "Policy Termination Date",
        "First Payment From Date",  # lag between days taken for first payment since received date
    ]
    NUMERIC = [
        "Insured Age at Loss",
        "Insured Annualized Salary",
        "Duration Months",
        "Policy Lives",
    ]
    CATEGORICAL = [
        "Insured Gender",
        "TSA Ind",
        "SS Pri Status",
        "SS Adjustment Ind",
        "Pre-Ex Outcome",
        "Insured Salary Ind",
        "Primary Diagnosis Category",
        "DOT Exertion Level (Primary)",
        "SS Pri Award Amt",
        "Coverage Code",
        "SIC Code",
    ]

    COLUMNS = (
        ID
        + DATES
        + NUMERIC
        + CATEGORICAL
        + ["Primary Diagnosis Code", "Primary Diagnosis Desc"]
    )

    settlement_df = df[COLUMNS].copy()
    settlement_df = resolve_formatting(settlement_df, DATES, NUMERIC)

    return settlement_df


def clean_pd_category(df):
    PD_CAT = [
        "disease_eye_adnexa",
        "disease_circulatory_system",
        "disease_ear_mastoid",
        "disease_respiratory_system",
        "disease_musculoskeletal",
        "injury_poisonining",
        "disease_mental_neuro",
        "neoplasms",
    ]
    PRIMARY_DIAG_CAT = "Primary Diagnosis Category"
    df.loc[:, PRIMARY_DIAG_CAT] = df.loc[:, PRIMARY_DIAG_CAT].str.lower()
    df.loc[df[PRIMARY_DIAG_CAT] == "", PRIMARY_DIAG_CAT] = "unknown"
    df.loc[df[PRIMARY_DIAG_CAT].isnull(), PRIMARY_DIAG_CAT] = "unknown"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith("diseases of the circulatory system"),
        PRIMARY_DIAG_CAT,
    ] = "disease_circulatory_system"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith("diseases of the eye & adnexa"),
        PRIMARY_DIAG_CAT,
    ] = "disease_eye_adnexa"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith("diseases of the ear & mastoid process"),
        PRIMARY_DIAG_CAT,
    ] = "disease_ear_mastoid"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith(
            "diseases of the musculoskeletal system & connectiv"
        ),
        PRIMARY_DIAG_CAT,
    ] = "disease_respiratory_system"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith("diseases of the respiratory system"),
        PRIMARY_DIAG_CAT,
    ] = "disease_musculoskeletal"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith(
            "injury, poisoning & certain other consequences of"
        ),
        PRIMARY_DIAG_CAT,
    ] = "injury_poisonining"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith(
            "mental, behavioral & neurodevelopmental disorders"
        ),
        PRIMARY_DIAG_CAT,
    ] = "disease_mental_neuro"
    df.loc[
        df[PRIMARY_DIAG_CAT].str.startswith("neoplasms"), PRIMARY_DIAG_CAT
    ] = "neoplasms"
    df.loc[~df[PRIMARY_DIAG_CAT].isin(PD_CAT), PRIMARY_DIAG_CAT] = "others"

    return df


def map_categories(df):

    df = clean_pd_category(df)

    SIC_CAT_INC_LIST = [
        "59",
        "26",
        "15",
        "32",
        "64",
        "44",
        "30",
        "56",
        "67",
        "49",
        "92",
        "73",
        "79",
        "63",
        "27",
        "99",
        "51",
        "34",
        "82",
        "80",
    ]
    SALARY_IND_EX_LIST = ["BI-WEEKLY", "Unknown", "WEEKLY", "YEARLY"]
    SS_PRI_ST_EX = ["Appealed", "Appealed ALJ", "Not Applied", "Pending", "Unknown"]

    COVERAGE_CODE_EX = ["LTDBU", "LTDCORE", "LTDVOL"]
    PD_1_VERY_HIGH_CHANCE = [
        "H35",
        "N30",
        "R29",
        "S13",
        "S33",
        "M96",
        "G60",
        "I73",
        "K50",
        "S92",
        "M06",
        "M41",
        "M50",
        "M60",
        "M46",
        "G82",
        "G83",
        "H81",
        "H90",
        "I89",
        "M24",
        "M72",
        "R53",
        "S12",
        "S52",
        "S89",
    ]
    PD_1_HIGH_CHANCE = [
        "G90",
        "I67",
        "M15",
        "M47",
        "M51",
        "M54",
        "S93",
        "B20",
        "G56",
        "G81",
        "G89",
        "H47",
        "I47",
        "M62",
        "M87",
        "R00",
        "S14",
        "S73",
    ]
    PD_1_MED_CHANCE = [
        "G43",
        "M48",
        "S82",
        "G61",
        "J45",
        "S32",
        "Z51",
        "G71",
        "G93",
        "H33",
        "L40",
        "M35",
        "M43",
        "E11",
        "F07",
        "G44",
        "G95",
        "G40",
        "C54",
        "C92",
    ]
    PD_1_COMBINED = PD_1_VERY_HIGH_CHANCE + PD_1_HIGH_CHANCE + PD_1_MED_CHANCE

    PD_2_VERY_HIGH = [
        "5XXA",
        "009",
        "030A",
        "14",
        "201A",
        "47",
        "898",
        "90XA",
        "019",
        "36",
        "009A",
        "06",
    ]

    PD_2_HIGH = [
        "34",
        "100",
        "001",
        "059",
        "119",
        "239",
        "39",
        "51",
        "909",
        "92XA",
        "09",
        "409A",
        "80",
        "109A",
        "30",
        "82",
        "5",
        "111",
        "13",
        "311",
        "354",
        "52",
        "659",
        "812",
    ]

    PD_2_MED = [
        "20",
        "26",
        "061",
        "109",
        "609",
        "817",
        "89",
        "12",
        "50",
        "219A",
        "529",
        "901",
        "9XXA",
        "00",
        "04",
        "02",
        "4",
    ]
    PD_2_COMBINED = PD_2_VERY_HIGH + PD_2_HIGH + PD_2_MED

    PRI_0_1000 = ["(0-250)", "(250-500)", "(500-750)", "(750-1000)"]
    PRI_1000_1250 = ["(1000-1250)"]
    PRI_1250_1500 = ["(1250-1500)"]
    PRI_1500_1750 = ["(1500-1750)"]
    PRI_1750_3000 = [
        "(1750-2000)",
        "(2000-2250)",
        "(2250-2500)",
        "(2500-2750)",
        "(2750-3000)",
    ]
    PRI_OTHERS = ["(3000-3250)", "NA"]

    df["SIC Category"] = df["SIC Code"].astype(str).apply(lambda x: x[:2])
    df.loc[~df["SIC Category"].isin(SIC_CAT_INC_LIST), "SIC Category"] = "others"

    df.loc[
        df["Insured Salary Ind"].isin(SALARY_IND_EX_LIST), "Insured Salary Ind"
    ] = "others"
    df.loc[df["SS Pri Status"].isin(SS_PRI_ST_EX), "SS Pri Status"] = "others"

    df.loc[df["Coverage Code"].isin(COVERAGE_CODE_EX), "Coverage Code"] = "LTDVOL"

    df = tokenize_pd_code(df)

    df.loc[:, "SS Pri Award Amt"] = np.where(
        (df["SS Pri Award Amt"].isnull()) | (df["SS Pri Award Amt"] == ""),
        "NA",
        df["SS Pri Award Amt"].astype(str).apply(lambda x: x[1:]),
    )
    df.loc[df["SS Pri Award Amt"].isin(PRI_0_1000), "SS Pri Award Amt"] = "(0-1000)"
    df.loc[
        df["SS Pri Award Amt"].isin(PRI_1000_1250), "SS Pri Award Amt"
    ] = "(1000-1250)"
    df.loc[
        df["SS Pri Award Amt"].isin(PRI_1250_1500), "SS Pri Award Amt"
    ] = "(1250-1500)"
    df.loc[
        df["SS Pri Award Amt"].isin(PRI_1500_1750), "SS Pri Award Amt"
    ] = "(1500-1750)"
    df.loc[
        df["SS Pri Award Amt"].isin(PRI_1750_3000), "SS Pri Award Amt"
    ] = "(1750-3000)"
    df.loc[df["SS Pri Award Amt"].isin(PRI_OTHERS), "SS Pri Award Amt"] = "others"

    df.loc[df["pd_code_1"].isin(PD_1_VERY_HIGH_CHANCE), "pd_1_cat"] = "very_high"
    df.loc[df["pd_code_1"].isin(PD_1_HIGH_CHANCE), "pd_1_cat"] = "high"
    df.loc[df["pd_code_1"].isin(PD_1_MED_CHANCE), "pd_1_cat"] = "medium"
    df.loc[~df["pd_code_1"].isin(PD_1_COMBINED), "pd_1_cat"] = "others"

    df.loc[df["pd_code_2"].isin(PD_2_VERY_HIGH), "pd_2_cat"] = "very_high"
    df.loc[df["pd_code_2"].isin(PD_2_HIGH), "pd_2_cat"] = "high"
    df.loc[df["pd_code_2"].isin(PD_2_MED), "pd_2_cat"] = "medium"
    df.loc[~df["pd_code_2"].isin(PD_2_COMBINED), "pd_2_cat"] = "others"

    INTER_DROP = [
        "index",
        "pd_code_1",
        "pd_code_2",
        "DOT Exertion Level (Primary)",
        "Primary Diagnosis Code",
        "Insured Salary Ind",
        "days_to_first_payment",
        "emp_tenure",
        "SIC Code",
    ]

    df.drop(INTER_DROP, axis=1, inplace=True)

    return df


def pre_process_pd_desc(df):
    PRIMARY_DIAG_DESC = "Primary Diagnosis Desc"

    df[PRIMARY_DIAG_DESC] = df[PRIMARY_DIAG_DESC].fillna("__na__").str.lower()
    inter_vertebral_dis = list(df[PRIMARY_DIAG_DESC].str.contains("intervertebral dis"))
    inter_vertebral_dis = [1 if x else 0 for x in inter_vertebral_dis]
    df.loc[:, "medical_cond_comb_intervertebral disease"] = inter_vertebral_dis

    cerebrovascular_disease = list(
        df[PRIMARY_DIAG_DESC].str.contains("cerebrovascular")
    )
    cerebrovascular_disease = [1 if x else 0 for x in cerebrovascular_disease]
    df.loc[:, "medical_cond_comb_cerebrovascular disease"] = cerebrovascular_disease

    fracture = list(df[PRIMARY_DIAG_DESC].str.contains("fracture"))
    fracture = [1 if x else 0 for x in fracture]
    df.loc[:, "medical_cond_comb_fracture"] = fracture

    acetabulum_disorder = list(df[PRIMARY_DIAG_DESC].str.contains("acetabulum"))
    acetabulum_disorder = [1 if x else 0 for x in acetabulum_disorder]
    df.loc[:, "medical_cond_comb_acetabulum disorder"] = acetabulum_disorder

    spondylosis = list(df[PRIMARY_DIAG_DESC].str.contains("spondylo"))
    spondylosis = [1 if x else 0 for x in spondylosis]
    df.loc[:, "medical_cond_comb_spondylosis"] = spondylosis

    sprains = list(df[PRIMARY_DIAG_DESC].str.contains("sprain"))
    sprains = [1 if x else 0 for x in sprains]
    df.loc[:, "medical_cond_comb_sprain"] = sprains

    localization = list(df[PRIMARY_DIAG_DESC].str.contains("localization"))
    localization = [1 if x else 0 for x in localization]
    df.loc[:, "medical_cond_comb_localization related"] = localization

    pain = list(df[PRIMARY_DIAG_DESC].str.contains("pain"))
    pain = [1 if x else 0 for x in pain]
    df.loc[:, "medical_cond_comb_pain"] = pain

    cervix = list(df[PRIMARY_DIAG_DESC].str.contains("cervi"))
    cervix = [1 if x else 0 for x in cervix]
    df.loc[:, "sys_organ_comb_cervical"] = cervix

    ligaments = list(df[PRIMARY_DIAG_DESC].str.contains("ligament"))
    ligaments = [1 if x else 0 for x in ligaments]
    df.loc[:, "sys_organ_comb_ligament"] = ligaments

    lumbosacral = list(df[PRIMARY_DIAG_DESC].str.contains("lumbosacral"))
    lumbosacral = [1 if x else 0 for x in lumbosacral]
    df.loc[:, "sys_organ_comb_lumbosacral"] = lumbosacral

    ankle = list(df[PRIMARY_DIAG_DESC].str.contains("ankle"))
    ankle = [1 if x else 0 for x in ankle]
    df.loc[:, "sys_organ_comb_ankle"] = ankle

    thoracic_region = list(
        (df[PRIMARY_DIAG_DESC].str.contains("thorac"))
        | (df[PRIMARY_DIAG_DESC].str.contains("thorax"))
    )
    thoracic_region = [1 if x else 0 for x in thoracic_region]
    df.loc[:, "sys_organ_comb_thoracic region"] = thoracic_region

    return df


def map_diag_entities(df):
    df = pre_process_pd_desc(df)
    MEDICAL_COND_LIST = [
        "hematuria",
        "dorsalgia",
        "tibial tendinitis",
        "idiopathic scoliosis",
        "laceration",
        "bucket-handle tear",
        "myositis",
        "concussion",
        "inflammatory demyelinating polyneuritis",
        "peripheral vascular disease",
        "migraine",
        "dislocation",
        "postlaminectomy syndrome",
        "back pain",
        "hemiplegia",
        "cervical disc displacement",
        "rheumatoid arthritis",
        "type 2 diabetes mellitus",
        "cervical disc disorders",
        "calcific tendinitis",
        "vascular headache",
        "crohn's disease",
        "guillain-barre syndrome",
        "carpal tunnel syndrome",
        "ischemic optic neuropathy",
        "cervical disc disorder",
        "lymphedema",
        "meniere's disease",
        "malignant neoplasm of endometrium",
        "myeloblastic leukemia",
        "postconcussional syndrome",
        "paraplegia",
        "asthma",
        "pigmentary retinal dystrophy",
        "human immunodeficiency virus",
        "polyosteoarthritis",
        "proliferative diabet",
        "syringomyelia",
        "sciatica",
        "tachycardia",
        "spinal stenosis",
        "radiculopathy",
        "cervical disc degeneration",
        "epilepsy",
        "arthritis",
        "muscular dystrophy",
        "rupture",
    ]  #'hereditary and idiopathic neuropathies', 'fatigue'
    SYS_ORGAN_LIST = [
        "acetabulum",
        "back",
        "ear",
        "intervertebral disc",
        "eye",
        "bilateral",
        "ulnar nerve",
        "upper",
        "wrist",
        "joint",
        "leg",
        "limb",
        "hip",
        "endometrium",
        "spinal cord",
        "lumbar region",
        "rotator cuff",
        "calcaneus",
        "pelvis",
        "shaft of right tibia",
        "spleen",
        "peripheral",
        "shoulder",
    ]
    df["Primary Diagnosis Desc"] = (
        df["Primary Diagnosis Desc"].fillna("__na__").str.lower()
    )

    cond_1_list = [
        [1 if x in desc else 0 for desc in list(df["Primary Diagnosis Desc"])]
        for x in MEDICAL_COND_LIST
    ]
    medical_cond_df = pd.DataFrame(cond_1_list).T
    medical_cond_df.columns = ["medical_cond_comb_" + x for x in MEDICAL_COND_LIST]
    medical_cond_df.loc[:, "medical_cond_comb_others"] = 0
    medical_cond_df.loc[
        medical_cond_df.sum(axis=1) == 0, "medical_cond_comb_others"
    ] = 1

    organ_list = [
        [1 if x in desc else 0 for desc in list(df["Primary Diagnosis Desc"])]
        for x in SYS_ORGAN_LIST
    ]
    sys_organ_df = pd.DataFrame(organ_list).T
    sys_organ_df.columns = ["sys_organ_comb_" + x for x in SYS_ORGAN_LIST]
    sys_organ_df.loc[:, "sys_organ_comb_others"] = 0
    sys_organ_df.loc[sys_organ_df.sum(axis=1) == 0, "sys_organ_comb_others"] = 1
    concat_df = pd.concat([df, medical_cond_df, sys_organ_df], axis=1)

    return concat_df


def scale_features(df, scaler):

    COLUMNS_SCALE = [
        "policy_tenure",
        "days_to_report",
        "emp_tenure",
        "prognosis_days",
        "days_to_first_payment",
        "Insured Annualized Salary",
        "Insured Age at Loss",
        "Policy Lives",
    ]
    df.loc[:, COLUMNS_SCALE] = scaler.transform(df.loc[:, COLUMNS_SCALE].to_numpy())
    return df


def posterior_correction(p1_orig, p_1_train, pred):

    cf = p1_orig / p_1_train
    ncf = (1 - p1_orig) / (1 - p_1_train)
    pcorr = (pred * cf) / (pred * cf + (1 - pred) * ncf)
    return pcorr


def test_train_match(train_template, test_data):
    missing_cols = set(train_template.columns) - set(test_data.columns)
    for c in missing_cols:
        test_data[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test_data = test_data[train_template.columns].copy()
    return test_data


def get_na_rows(test_data):

    columns = [
        "Insured Gender",
        "TSA Ind",
        "SS Pri Status",
        "SS Adjustment Ind",
        "Pre-Ex Outcome",
        "SS Pri Award Amt",
        "Primary Diagnosis Category",
        "Coverage Code",
        "SIC Code",
        "Primary Diagnosis Code",
        "Insured Age at Loss",
        "Insured Annualized Salary",
        "Policy Lives",
        "policy_tenure",
        "days_to_report",
        "prognosis_days",
        "Primary Diagnosis Desc",
    ]

    col_locs = [test_data.columns.get_loc(col) for col in columns]

    na_inds = []
    for _, row in test_data.iterrows():
        na_inds.append(row[col_locs].isnull().any())
    test_data.loc[:, "NA_row"] = ["Y" if x else "N" for x in na_inds]
    return test_data


def get_bucket_and_key_from_s3_uri(uri):
    bucket, key = uri.split("/", 2)[-1].split("/", 1)
    return bucket, key


def download_obj_from_s3(bucket_name, key, artifact_type):
    """
    NamedTemporaryFile works for the model artifacts.
    Scaler needs to be downloaded to local file system and loaded for inference
    """

    bucket = boto3.resource("s3").Bucket(bucket_name)

    if artifact_type == "dl_model":
        with tempfile.NamedTemporaryFile(suffix=".h5") as fp:
            bucket.download_fileobj(key, fp)
            return onnxruntime.InferenceSession(fp.name)

    elif artifact_type == "xgb_model":
        with tempfile.NamedTemporaryFile(suffix=".joblib") as fp:
            bucket.download_fileobj(key, fp)
            return joblib.load(fp.name)
    else:
        with tempfile.TemporaryDirectory() as dirpath:
            # tempfile throws an EOFError for scaler. Use normal file inside temp directory
            with open(f"{dirpath}/scaler.joblib", "wb") as file_data:
                bucket.download_fileobj(key, file_data)
            file_data.close()
            return joblib.load(f"{dirpath}/scaler.joblib")
