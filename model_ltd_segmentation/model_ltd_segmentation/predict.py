def predict(**kwargs):
    import boto3
    import joblib
    import pandas as pd
    import numpy as np
    from ordered_set import OrderedSet
    from itertools import compress
    from torchtext.data.utils import get_tokenizer
    from torch.nn.utils.rnn import pad_sequence
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    MODEL_BUCKET = "spr-barrel-models"
    MODEL_DIR = "RSLI-Segmentation"
    GNEWS_EMBEDDING = f"{MODEL_DIR}/embeddings/embedding_matrix_gnews_300d.pkl"
    GLOVE_EMBEDDING = f"{MODEL_DIR}/embeddings/embedding_matrix_glove_300d.pkl"
    CUSTOM_EMBEDDING = (
        f"{MODEL_DIR}/embeddings/embedding_matrix_desc_300d_2023_11_11.pkl"
    )
    PD_CODE_EMBEDDING = (
        f"{MODEL_DIR}/embeddings/embedding_matrix_code_300d_2023_11_11.pkl"
    )
    ROBUST_SCALER = f"{MODEL_DIR}/scalers/scaler_obj_2023_11_11.pkl"
    PD_DESC_TOKENIZER = f"{MODEL_DIR}/tokenizers/tokenizer_desc_2023_11_11.pkl"
    PD_CODE_TOKENIZER = f"{MODEL_DIR}/tokenizers/tokenizer_code_2023_11_11.pkl"
    MODEL_1_KEY = f"{MODEL_DIR}/main_model/torch_model_custom_2023_11_11.pth"
    MODEL_2_KEY = f"{MODEL_DIR}/main_model/torch_model_gnews_2023_11_11.pth"
    MODEL_3_KEY = f"{MODEL_DIR}/main_model/torch_model_glove_2023_11_11.pth"

    PD_DESC_MAXLENGTH = 10
    PD_CODE_MAXLENGTH = 4

    CUSTOM_MODEL_WEIGHT = 0.3
    GLOVE_MODEL_WEIGHT = 0.3
    GNEWS_MODEL_WEIGHT = 0.4

    PRIMARY_FEATS = [
        "Insured Age at Loss",
        "Primary Diagnosis Category",
        "DOT Exertion Level (Primary)",
        "Insured Gender",
        "Occ Category",
        "Any Occ Category",
        "SIC Category",
        "STD to LTD Bridge Ind",
        "Eligibility Outcome",
        "SS Primary Hardship Indicator",
        "TSA Ind",
        "Mental Nervous Ind",
        "Insured Salary Ind",
        "Net Benefit",
        "prognosis_days",
        "Policy Lives",
    ]

    REDUNDANT_FEATS = [
        "Primary Diagnosis Category_UNKNOWN",
        "Primary Diagnosis Category_UNSPECIFIED",
        "DOT Exertion Level (Primary)_?",
        "Insured Gender_M",
        "Occ Category_Dependent",
        "Any Occ Category_Own Occ Only",
        "SIC Category_Unknown",
        "Eligibility Outcome_PENDING",
        "SS Primary Hardship Indicator_N",
        "TSA Ind_N",
        "STD to LTD Bridge Ind_No Bridge",
        "Insured Salary Ind_W",
        "Mental Nervous Ind_3",
        "Mental Nervous Ind_5",
        "Mental Nervous Ind_EXCLUDED",
    ]

    COLS_TO_SCALE = [
        "prognosis_days",
        "Insured Age at Loss",
        "Net Benefit",
        "Policy Lives",
    ]

    def download_file_from_s3(bucket_name, key, local_file_name):
        s3 = boto3.client("s3")
        s3.download_file(bucket_name, key, local_file_name)

    def download_obj_from_s3(
        bucket_name, key, local_file_name, obj_type, load_local=False
    ):
        if not load_local:
            download_file_from_s3(bucket_name, key, str(local_file_name))
        if obj_type == "joblib":
            loaded_obj = joblib.load(local_file_name)
            return loaded_obj
        elif obj_type == "torch":
            model = torch.load(str(local_file_name))
            model.eval()
            return model

    def reconst_diag_codes(code_list):
        """
        Extract diagnosis codes at even positions from the ';' delimited claim diagnoses (cd) field.
        Join them back to reconstruct the expanded primary diagnosis code.

        Args:
            code_list (List): A list containing all the diagnosis codes and diagnosis description

        Returns:
            List: containing just diagnosis code delimited by '.'

        """

        list1 = [code_list[i] if i < len(code_list) else -1 for i in [0, 2, 4]]
        mask = [ele != -1 for ele in list1]
        pd_codes = list(OrderedSet(compress(list1, mask)))
        return ".".join(pd_codes)

    def int_reconstruct(df, token_column):
        """
            function to convert diag codes with leading zeros to ints so the vocabulary stays compact
        Args:
            df (pd.Dataframe): Dataset containing the column
            token_column (str): target column

        Returns:
            pd.Dataframe: Dataset containing modified column

        """
        pd_tokens_list = []

        for i in range(df.shape[0]):
            pd_tokens = [
                str(int(ele)) if ele.isdigit() else ele
                for ele in df.loc[i, token_column]
            ]
            pd_tokens_list.append(pd_tokens)

        return pd_tokens_list

    def claim_diagosis_clean(df, diagnosis_field="Claim Diagnoses"):
        """
        Removes blank tokens from the primary diagnosis field field a;;b;c --> a;b;c

        Args:
            df (pd.DataFrame): Dataset containing diagnosis field
            diagnosis_field (str): String containing the column identifier

        Returns:
            pd.DataFrame: Dataset containing reconstructed diagnosis field

        """

        df.loc[:, diagnosis_field] = (
            df.loc[:, diagnosis_field].fillna("-1").astype(str).values
        )
        df.loc[:, "cd_token"] = (
            df.loc[:, diagnosis_field]
            .apply(lambda x: [ele.strip() for ele in x.split(";")])
            .values
        )
        df.loc[:, "cd_token"] = df.loc[:, "cd_token"].apply(
            lambda x: [ele for ele in x if ele != ""]
        )

        df.loc[:, "primary_diag_reconstructed"] = df.loc[:, "cd_token"].apply(
            reconst_diag_codes
        )
        df.loc[:, "primary_diag_reconstructed"] = (
            df.loc[:, "primary_diag_reconstructed"].str.lower().values
        )

        df.loc[:, "pd_token"] = df.loc[:, "primary_diag_reconstructed"].apply(
            lambda x: x.split(".")
        )

        df = df.reset_index(drop=True, inplace=False)

        df.loc[:, "pd_token_mod"] = int_reconstruct(df, "pd_token")
        df.loc[:, "pd_reconstructed"] = df.loc[:, "pd_token_mod"].apply(
            lambda x: ".".join(x)
        )

        return df

    def generate_vectors(tokenizer, vocab, data, field, maxlen):
        """
        convert words to vectors and pads them

        Args:
            tokenizer (torchtext.tokenizer): pytorch tokenizer
            vocab (torchtext.Vocab): Vocab object
            data (pd.DataFrame): Dataset containing text field
            field (str): Text field identifier
            maxlen (int): maximum length for padding

        Returns:
            torch.tensor: tensor containing padded word vectors

        """
        tensor_list = []
        for seq_ in data[field]:
            tensor_list.append(
                torch.tensor(
                    [vocab[token] for token in tokenizer(seq_)], dtype=torch.long
                )[:maxlen]
            )
        return pad_sequence(tensor_list, padding_value=vocab["<pad>"], batch_first=True)

    def match_template(df, df_template):
        columns_missing = list(set(df_template.columns) - set(df.columns))
        df[[columns_missing]] = 0
        assert (
            df.shape[1] == df_template.shape[1]
        ), f"DataFrame shape is {df.shape}, expected X,{df_template.shape[1]}"
        return df

    def pre_process_statistical_feats(df):
        df.loc[:, "Insured Age at Loss"] = pd.to_numeric(
            df.loc[:, "Insured Age at Loss"]
        ).fillna(54)
        df.loc[:, "Insured Annualized Salary"] = (
            pd.to_numeric(df.loc[:, "Insured Annualized Salary"].str.replace(",", ""))
            .fillna(4.773800e04)
            .values
        )

        df["prognosis_days_1"] = pd.to_datetime(
            df["Potential Resolution Date"], errors="coerce"
        ) - pd.to_datetime(df["Loss Date"])
        df["prognosis_days_1"] = df["prognosis_days_1"] / np.timedelta64(1, "D")

        df["prognosis_days_2"] = pd.to_datetime(
            df["Duration Date"], errors="coerce"
        ) - pd.to_datetime(df["Loss Date"])
        df["prognosis_days_2"] = df["prognosis_days_2"] / np.timedelta64(1, "D")

        df.loc[:, "prognosis_days"] = df.loc[:, "prognosis_days_1"]
        df.loc[
            (df["prognosis_days"].isnull()) | (df["prognosis_days"] < 0),
            "prognosis_days",
        ] = df.loc[df["prognosis_days"].isnull(), "prognosis_days_2"]

        df.loc[:, "prognosis_days"] = df.loc[:, "prognosis_days"].fillna(1276)

        df.loc[:, "Policy Lives"] = (
            pd.to_numeric(df.loc[:, "Policy Lives"].str.replace(",", ""))
            .fillna(1899)
            .values
        )
        df.loc[:, "Net Benefit"] = (
            pd.to_numeric(df.loc[:, "Net Benefit"].str.replace(",", ""))
            .fillna(798)
            .values
        )
        df.loc[:, "Annualized Premium"] = (
            pd.to_numeric(df.loc[:, "Annualized Premium"].str.replace(",", ""))
            .fillna(3.525500e05)
            .values
        )

        df.loc[:, "SS Pursue Ind"] = df.loc[:, "SS Pursue Ind"].apply(
            lambda x: "YES" if x == "Yes" else x
        )
        df.loc[:, "Mental Nervous Ind"] = df.loc[:, "Mental Nervous Ind"].apply(
            lambda x: "EXCLUDED" if x == "Excluded" else x
        )
        df.loc[:, "Mental Nervous Ind"] = df.loc[:, "Mental Nervous Ind"].apply(
            lambda x: "FULL" if x == "Full" else x
        )
        df.loc[:, "Mental Nervous Ind"] = df.loc[:, "Mental Nervous Ind"].apply(
            lambda x: "FULL" if x == "Full" else x
        )
        return df

    def pre_process_text(df):
        # Replace "na" with an empty string
        df["Primary Diagnosis Desc"].replace("na", "", inplace=True)

        # Remove non-alphanumeric characters and replace '-' with space
        df["Primary Diagnosis Desc"] = (
            df["Primary Diagnosis Desc"]
            .str.replace(r"[^\w\s\-]", "")
            .str.replace(r"-", " ")
        )

        # Remove extra whitespaces
        df["Primary Diagnosis Desc"] = df["Primary Diagnosis Desc"].apply(
            lambda x: " ".join(str(x).split())
        )

        # Replace "ms" with "multiple sclerosis"
        df["Primary Diagnosis Desc"] = df["Primary Diagnosis Desc"].str.replace(
            r"\bms\b", "multiple sclerosis"
        )
        df.loc[:, "Primary Diagnosis Desc"] = df["Primary Diagnosis Desc"].fillna(
            "_na_"
        )
        df.loc[:, "pd_reconstructed"] = df["pd_reconstructed"].astype(str)
        return df

    class SegmentationModel(nn.Module):
        def __init__(
            self,
            embedding_matrix_desc,
            pd_vocab_size,
            pd_embed_size,
            pd_embedding_matrix,
            vocab_size_desc,
            embed_size_desc,
            stat_input_size,
            train_desc_flag=False,
        ):
            super().__init__()

            # Text input
            self.text_embedding = nn.Embedding(vocab_size_desc, embed_size_desc)
            self.text_embedding.weight = nn.Parameter(
                embedding_matrix_desc, requires_grad=train_desc_flag
            )
            self.text_lstm = nn.LSTM(
                embed_size_desc, 128, batch_first=True, bidirectional=True
            )
            self.text_dropout = nn.Dropout(0.4)
            self.text_conv1d = nn.Conv1d(256, 32, kernel_size=2)
            self.text_maxpool1d = nn.AdaptiveMaxPool1d(1)
            self.text_dense = nn.Linear(32, 32)

            # Stat input
            self.stat_dense1 = nn.Linear(stat_input_size, 128)
            self.stat_batchnorm1 = nn.BatchNorm1d(128)
            self.stat_dense2 = nn.Linear(128, 64)
            self.stat_batchnorm2 = nn.BatchNorm1d(64)
            self.stat_dropout = nn.Dropout(0.3)

            # PD code input
            self.pd_embedding = nn.Embedding(pd_vocab_size, pd_embed_size)
            self.pd_embedding.weight = nn.Parameter(
                pd_embedding_matrix, requires_grad=False
            )
            self.pd_lstm = nn.LSTM(
                pd_embed_size, 128, batch_first=True, bidirectional=True
            )
            self.pd_batchnorm = nn.BatchNorm1d(num_features=4)
            self.pd_dropout = nn.Dropout(0.4)
            self.pd_conv1d = nn.Conv1d(256, 32, kernel_size=2)
            self.pd_maxpool1d = nn.AdaptiveMaxPool1d(1)
            self.pd_dense1 = nn.Linear(32, 32)
            self.pd_dense2 = nn.Linear(32, 32)

            # Merged layers
            self.merged_dense1 = nn.Linear(128, 128)
            self.merged_dropout = nn.Dropout(0.2)
            self.merged_batchnorm = nn.BatchNorm1d(128)
            self.merged_dense2 = nn.Linear(128, 128)

            # Output layer
            self.out = nn.Linear(128, 5)

        def forward(self, text_input, stat_input, pd_input):
            # Text input
            x1 = self.text_embedding(text_input)
            x1, _ = self.text_lstm(x1)
            x1 = self.text_dropout(x1)
            x1 = self.text_conv1d(x1.permute(0, 2, 1))
            x1 = self.text_maxpool1d(x1)
            x1 = self.text_dense(x1.view(-1, 32))
            x1 = nn.ReLU()(x1)

            # Stat input
            x2 = self.stat_dense1(stat_input)
            x2 = nn.ReLU()(x2)
            x2 = self.stat_batchnorm1(x2)
            x2 = self.stat_dense2(x2)
            x2 = nn.ReLU()(x2)
            x2 = self.stat_batchnorm2(x2)
            x2 = self.stat_dropout(x2)

            # PD code input
            pd_x = self.pd_embedding(pd_input)
            pd_x, _ = self.pd_lstm(pd_x)
            pd_x = self.pd_batchnorm(pd_x)
            pd_x = self.pd_dropout(pd_x)
            pd_x = self.pd_conv1d(pd_x.permute(0, 2, 1))
            pd_x = self.pd_maxpool1d(pd_x)
            pd_x = self.pd_dense1(pd_x.view(-1, 32))
            pd_x = nn.ReLU()(pd_x)
            pd_x = self.pd_dense2(pd_x)
            pd_x = nn.ReLU()(pd_x)

            # Merge
            merged_layer = torch.cat([x1, x2, pd_x], dim=1)

            # Common layers
            x = self.merged_dense1(merged_layer)
            x = nn.ReLU()(x)
            x = self.merged_dropout(x)
            x = self.merged_batchnorm(x)
            x = self.merged_dense2(x)
            x = nn.ReLU()(x)

            # Output layer
            out = self.out(x)
            out = F.softmax(out, dim=1)

            return out

    # Downloading artifacts
    embedding_matrix_gnews = download_obj_from_s3(
        MODEL_BUCKET,
        GNEWS_EMBEDDING,
        "data/embedding_matrix_gnews_2023_11_11.pkl",
        "joblib",
    )
    embedding_matrix_glove = download_obj_from_s3(
        MODEL_BUCKET,
        GLOVE_EMBEDDING,
        "data/embedding_matrix_glove_2023_11_11.pkl",
        "joblib",
    )
    embedding_matrix_custom_desc = download_obj_from_s3(
        MODEL_BUCKET,
        CUSTOM_EMBEDDING,
        "data/embedding_matrix_desc_2023_11_11.pkl",
        "joblib",
    )
    embedding_matrix_custom_code = download_obj_from_s3(
        MODEL_BUCKET,
        PD_CODE_EMBEDDING,
        "data/embedding_matrix_code_2023_11_11.pkl",
        "joblib",
    )
    sc = download_obj_from_s3(
        MODEL_BUCKET, ROBUST_SCALER, "data/scaler_obj_2023_11_11.pkl", "joblib"
    )
    pd_desc_vocab_obj = download_obj_from_s3(
        MODEL_BUCKET,
        PD_DESC_TOKENIZER,
        "data/tokenizer_desc_2023_11_11.pkl",
        "joblib",
    )
    pd_code_vocab_obj = download_obj_from_s3(
        MODEL_BUCKET,
        PD_CODE_TOKENIZER,
        "data/tokenizer_code_2023_11_11.pkl",
        "joblib",
    )
    # Init models
    globals()[
        "SegmentationModel"
    ] = SegmentationModel  # torch load expects model definition to be defined in main namespace
    model_1 = download_obj_from_s3(
        MODEL_BUCKET,
        MODEL_1_KEY,
        "data/temp_model1.pth",
        "torch",
    )

    model_2 = download_obj_from_s3(
        MODEL_BUCKET,
        MODEL_2_KEY,
        "data/temp_model2.pth",
        "torch",
    )

    model_3 = download_obj_from_s3(
        MODEL_BUCKET,
        MODEL_3_KEY,
        "data/temp_model3.pth",
        "torch",
    )
    # Init templates
    df_template = pd.read_csv(
        "data/data_template.csv", engine="pyarrow", dtype_backend="pyarrow"
    )
    dtype_template = download_obj_from_s3(
        "", "", "data/dtype_template.pkl", "joblib", True
    )
    # Init tokenizers
    tokenizer_pd_desc = get_tokenizer("basic_english")
    tokenizer_pd_code = get_tokenizer(lambda x: x.split("."))

    # Converting Dtypes
    embedding_matrix_gnews = torch.FloatTensor(embedding_matrix_gnews)
    embedding_matrix_glove = torch.FloatTensor(embedding_matrix_glove)
    embedding_matrix_custom_desc = torch.FloatTensor(embedding_matrix_custom_desc)
    embedding_matrix_custom_code = torch.FloatTensor(embedding_matrix_custom_code)

    df = pd.DataFrame([kwargs.get("inputs").get("claim")])
    # Processing claim diagnosis code
    df = claim_diagosis_clean(df, "Claim Diagnoses")
    # Filter group 2
    df = pre_process_statistical_feats(df)
    df_sub = df.loc[:, PRIMARY_FEATS]
    df_sub = df_sub.replace(r"^\s*$", np.nan, regex=True)
    # Generating numerical Features
    df_sub_dummies = pd.get_dummies(df_sub)
    df_sub_dummies = df_sub_dummies.drop(REDUNDANT_FEATS, axis=1, errors="ignore")
    # preprocessing text features
    df_sub_dummies["Primary Diagnosis Desc"] = df["Primary Diagnosis Desc"].str.lower()
    df_sub_dummies["pd_reconstructed"] = df["pd_reconstructed"].str.lower()
    df_base = pre_process_text(df_sub_dummies)
    # Scaling Features
    df_base.loc[:, COLS_TO_SCALE] = sc.transform(
        df_base.loc[:, COLS_TO_SCALE].to_numpy()
    )
    # Separating text inputs from numerical inputs
    primary_diagnosis_df = df_base[["Primary Diagnosis Desc"]]
    pd_code_df = df_base[["pd_reconstructed"]]
    df_base = df_base.drop(
        ["pd_reconstructed", "Primary Diagnosis Desc"],
        inplace=False,
        axis=1,
    )
    # Matching train template
    df_base = match_template(df_base, df_template)
    # Setting dtypes
    df_base = df_base.astype(dtype_template)
    # Converting numerical values to tensors
    np_array = df_base.values
    if np_array.dtype == object:
        np_array = np_array.astype(float)
    df_base = torch.tensor(np_array, dtype=torch.float32)

    # Vectorizing text
    pd_desc = generate_vectors(
        tokenizer_pd_desc,
        pd_desc_vocab_obj,
        primary_diagnosis_df,
        "Primary Diagnosis Desc",
        PD_DESC_MAXLENGTH,
    )
    pd_code = generate_vectors(
        tokenizer_pd_code,
        pd_code_vocab_obj,
        pd_code_df,
        "pd_reconstructed",
        PD_CODE_MAXLENGTH,
    )
    assert (
        pd_desc.shape[1] == 10
    ), f"DataFrame shape is {pd_desc.shape}, expected X,{10}"
    assert pd_code.shape[1] == 4, f"DataFrame shape is {pd_code.shape}, expected X,{4}"

    # Init model param
    stat_dim = df_base.shape[1]
    pdesc_vocab_size = embedding_matrix_custom_desc.shape[0] + 1
    pdesc_embed_size = embedding_matrix_custom_desc.shape[1]
    pd_vocab_size = embedding_matrix_custom_code.shape[0] + 1
    pd_embed_size = embedding_matrix_custom_code.shape[1]

    # Prediction
    y_test_preds_custom = model_1(pd_desc, df_base, pd_code)
    y_test_preds_glove = model_2(pd_desc, df_base, pd_code)
    y_test_preds_gnews = model_3(pd_desc, df_base, pd_code)

    y_preds_combined = (
        CUSTOM_MODEL_WEIGHT * y_test_preds_custom
        + GNEWS_MODEL_WEIGHT * y_test_preds_gnews
        + GLOVE_MODEL_WEIGHT * y_test_preds_glove
    )
    y_preds_combined_class = np.argmax(y_preds_combined.detach().numpy(), axis=1)

    return y_preds_combined_class


if __name__ == "__main__":
    _ = predict()
