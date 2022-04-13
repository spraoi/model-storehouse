def predict(**kwargs):
    import joblib
    import numpy as np
    import pandas as pd
    import json
    from helpers import (
        preprocess,
        Diff,
        categorical_columns,
        date_columns,
        siu_feat_diff,
        ops_feat_diff,
        date_diff,
    )

    # loading data and templates
    df_1_loc = kwargs.get("inputs").get("dataset_loc")  # FIXME
    df_1_template_loc = kwargs.get("inputs").get("template_loc")
    df_1_select_loc = kwargs.get("inputs").get("select_loc")
    df_1_right_loc = kwargs.get("inputs").get("right_loc")
    df_1_left = pd.read_csv(df_1_loc, index_col=0)
    df_1_template = pd.read_csv(df_1_template_loc, index_col=0)
    df_1_select = pd.read_csv(df_1_select_loc, index_col=0)
    df_1_right = pd.read_csv(df_1_right_loc, index_col=0, low_memory=False)    

    # merge with 3 tables
    df_1 = df_1_left.merge(df_1_right, on="id", how="left")
    assert len(df_1)==len(df_1_left)

    # dropping date columns
    new_list = [x for x in date_columns if x not in date_diff]

    df_1.drop(columns=new_list, inplace=True, errors="ignore")

    assert df_1.type.nunique()==1 #checking if passed df is single type

    # preprocessing and loading designated model
    channel = "Issue"
    role = "ops"
    loaded_model = joblib.load("./artifacts/models/Issue_ops_v3.sav")
    # template matching
    x = df_1_select.columns.difference(df_1.columns).tolist()
    x.remove(role + ".actualDisposition")
    df_1[x] = "NaN"
    df_1.drop(columns=ops_feat_diff, inplace=True, errors="ignore")
    categorical_columns = Diff(categorical_columns, ops_feat_diff)
    categorical_columns = categorical_columns + ["modelType"]
    df = preprocess(df_1, categorical_columns, role=role)

    # verifying with template
    missing_cols = df_1_template.columns.difference(df.columns).tolist()
    missing_cols.remove("class")
    df[missing_cols] = "NaN"
    template_cols = df_1_template.columns.tolist()
    template_cols.remove("class")

    # checking for duplicated columns and template matching
    df_2 = df.loc[:, ~df.columns.duplicated()][template_cols]
    # NaN fixing
    df_2 = df_2.replace((np.inf, -np.inf, np.nan, "nan", "NaN"), 0).reset_index(
        drop=True
    )
    # index pop
    id = df_2.pop("shortId")
    # prediction dataframe
    df_pred = df_2.values

    # Inference step
    predictions_df = pd.DataFrame(
        loaded_model.predict(df_pred), columns=["predictions"]
    )
    predictions_df["shortId"] = id
    # remove set index to return shortId too as json load
    predictions_df = predictions_df.set_index("shortId")
    # save to local
    # predictions_df.to_csv("predictions.csv")
    # payload gen
    payload_json = json.loads(predictions_df.to_json(orient="records"))

    return [{"predicted_disposition": payload_json}]


# if __name__ == "__main__":
#     print(
#         predict(
#             inputs={
#                 "dataset_loc": "./artifacts/datasets/txn_employee_df_mod.csv",
#                 "template_loc": "./artifacts/datasets/transaction_siu_post_template_1.csv",
#                 "select_loc": "./artifacts/datasets/transaction_siu_select_template_1.csv",
#                 "right_loc": "./artifacts/datasets/triple_merge.csv"
#             }
#         )
#     )
