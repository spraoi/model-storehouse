def predict(**kwargs):

    import pandas as pd
    import shap
    import numpy as np
    import joblib
    import pkg_resources
    # from model_corie.model_resources.helpers import get_model_info
    # from model_corie.model_resources.helpers import fetch_artifact_key
    # from model_corie.model_resources.helpers import extract_top_drivers
    from utils import extract_top_drivers

    # config
    # s3_model_bucket = 'spr-ml-artifacts'
    # modelId = kwargs.get("inputs").get('modelId', None)
    # version_number = kwargs.get("inputs").get('version_number', None)

    #     'spr:bz:mod::9996c22e-f26c-425f-9a41-35865ebfc5f2'
    IP_NO_Levels = 60
    model_numeric_cols = [
        'DST_Distr_age', 'PH_Issue_age', 'DST_Distr_tenure', 'PD_APE',
        'PD_cover_amt'
    ]
    model_cat_cols = ['DST_Distr_typ', 'PD_Prod_code', 'PD_issue_qtr']
    #local files
    model_file = "corie_model_spr_bz__comp__aefb8f7c-d2a2-47a3-adde-12724ac2de78.joblib"
    template_file = "corie_template_data_spr_bz__comp__aefb8f7c-d2a2-47a3-adde-12724ac2de78.csv"
    test_x_file = "full_prediction_data_32k_118.csv"

#     test_dataset_id = kwargs.get("inputs").get("datasetId")
#     testing_data = load_dataframe(test_dataset_id).compute()
# using new ml-model-service
#     xgb_key = fetch_artifact_key(host, modelId, version_number, "model_key")
#     xgb_model = download_obj_from_s3(s3_model_bucket, xgb_key, "xgb_model.joblib")
    xgb_model_stream = pkg_resources.resource_stream("model_corie",
                                                     f"data/{model_file}")
    xgb_model = joblib.load(xgb_model_stream)

    template_fp = pkg_resources.resource_filename("model_corie",
                                                  f"data/{template_file}")
    template_data = pd.read_csv(template_fp)
    features = template_data.columns.values

    testX_fp = pkg_resources.resource_filename("model_corie",
                                               f"data/{test_x_file}")
    testing_data = pd.read_csv(testX_fp)
    #     template_key = fetch_artifact_key(host, modelId, version_number,
    #                                       "template_key")
    #     template_data = pd.read_csv(f"s3://{s3_model_bucket}/{template_key}")

    for col in model_numeric_cols:
        testing_data[col] = pd.to_numeric(testing_data[col])
    for col in model_cat_cols:
        testing_data[col] = testing_data[col].astype('category')

    def extract_features(test_data, features):
        # separate categorical  columns
        features_categorical = set(model_cat_cols)
        features_continuous = set(model_numeric_cols)
        variable_columns = features_categorical | features_continuous
        # Converting set to list
        features_categorical = list(features_categorical)
        features_continuous = list(features_continuous)
        variable_columns = list(variable_columns)
        # copying only feature columns
        test_data_final = test_data[variable_columns]
        cat_cols = [
            col for col in features_categorical
            if test_data_final[col].nunique() <= IP_NO_Levels
        ]
        test_data_final = pd.get_dummies(test_data_final, columns=cat_cols)
        test_data_init = test_data_final.copy()
        # Get missing columns in the training test
        missing_cols = set(features) - set(test_data_final.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            test_data_final[c] = 0
        # Ensure the order of column in the test set is
        # in the same order than in train set
        test_data_final = test_data_final[list(features)]
        X = test_data_final[test_data_final.columns].values

        return (X, test_data_final[test_data_final.columns], test_data_init,
                test_data)

    testing_data = testing_data.head(kwargs.get('nrows'))

    test_x, test_x_df, test_data_init, _ = extract_features(
        testing_data, features)
    predict_y = xgb_model.predict(test_x_df)

    def predict_fn(x):
        return xgb_model.predict_proba(x).astype(float)

    surr_prob = predict_fn(test_x_df)
    # shap explainer
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(test_x_df)
    shapley_df = extract_top_drivers(test_x_df, shap_values)
    # add predicted labels to the input df
    testing_data['Predicted_Status'] = np.where(predict_y == 1, 'Surrender',
                                                'Not_Surrender').ravel()
    testing_data['Surrender_Probability'] = surr_prob[:, 1]
    testing_data['NOT_Surrender_Probability'] = surr_prob[:, 0]
    testing_data['Churn_Score'] = np.round(
        testing_data['Surrender_Probability'] * 100, decimals=0)
    testing_data['Retention_Score'] = np.round(
        testing_data['NOT_Surrender_Probability'] * 100, decimals=0)
    testing_data['Surrender_Severity'] = pd.cut(
        testing_data['Churn_Score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical'],
        include_lowest=True)
    testing_data = testing_data.reset_index(drop=True)
    shapley_df = shapley_df.reset_index(drop=True)
    final_complete_df = pd.concat([testing_data, shapley_df], axis=1)
    return final_complete_df


print(predict(nrows=10).head())