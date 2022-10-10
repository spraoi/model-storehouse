def predict(**kwargs):
    import pandas as pd
    import numpy as np
    from inflection import humanize, underscore
    import joblib
    import pkg_resources
    import re
    import onnxruntime
    import logging

#     logging.basicConfig(level=logging.DEBUG)
    logging = logging.getLogger("airflow.task")


    dataset_id = kwargs.get("inputs").get("datasetId")
    logging.debug(f'{dataset_id}')

    if not kwargs.get("inputs").get("columns"):
        return [
            {
                "inputDataSource": f"{dataset_id}:0",
                "entityId": dataset_id,
                "predictedResult": [],
            }
        ]

    def padarray(A):
        t = 16 - len(A)
        return np.pad(A, pad_width=(t, 0), mode="constant")

    def read_objs(name):
        res_loc = pkg_resources.resource_stream("model_ffm", "data/" + name + ".joblib")
        # with open('data/' + name + '2.4.6.pkl', 'rb') as f:
        #     a = pickle.load(f)
        a = joblib.load(res_loc)
        logging.debug(f'{len(head_dict)}')
        return a

    def filter_bank(df):
        _mask = (df["ent"].isin(["UNK"])) & ~df["head"].isin(
            ["GroupNumber", "AccountNumber"]
        )
        _mask_1 = (df["ent"].isin(["Spouse", "Child", "Dependent"])) & (
            df["prod"].isin(["VIS", "DEN"])
        )
        _mask_9 = df["base"].isin(["Eligibility_DOH"])
        _mask_8 = (df["base"].isin(["Employee_Address"])) & (df["head"] == "Zip")
        _mask_10 = df["base"].isin(["Class/Set"])
        _mask_13 = (df["base"].isin(["Employee_Address_3"])) & (
            df["head"] == "AddressLine3"
        )
        _mask_14 = df["head"] == "GF_BenefitAmount"
        _mask_17 = (
            df["base"].isin(
                [
                    "Supplemental/Voluntary_Employee_Life_In_Force_Amount",
                    "Supplemental/Voluntary_Employee_Life_Applied_For_Amount",
                ]
            )
        ) & (df["prod"] == "LIFESUP")
        _mask_18 = (
            df["base"].isin(
                [
                    "Supplemental/Voluntary_Employee_AD&D_Applied_For_Amount",
                    "Supplemental/Voluntary_Employee_AD&D_In_Force_Amount",
                ]
            )
        ) & (df["prod"] == "ADDSUP")
        _mask_20 = (
            df["base"].isin(
                [
                    "EMPLOYEE_Voluntary/Supplemental_Coverage_LIFE_Benefit_Amount",
                    "EMPLOYEE_Voluntary/SupplementalCoverage_LIFE_Benefit_Amount",
                ]
            )
        ) & (df["prod"] == "LIFEVOL")
        _mask_21 = (
            df["base"].isin(["Dependent_Relationship_Dependent_Relationship"])
        ) & (df["ent"] == "Dependent")
        _mask_23 = (
            (df["head"].isin(["AccountNumber"]))
            & (df["ent"] == "Primary")
            & (df["prod"] == "UNK")
        )
        _mask_24 = (
            (
                df["base"].isin(
                    ["Basic_Life_In_Force_Amount", "Basic_Life_Applied_For_Amount"]
                )
            )
            & (df["ent"] == "Primary")
            & (df["prod"] == "LTD")
        )
        _mask_25 = (
            (df["base"].isin(["Supplemental/Voluntary_Employee_Life_Effective_Date"]))
            & (df["head"].isin(["EffectiveDate"]))
            & (df["prod"] == "LIFESUP")
        )
        _mask_27 = (
            (~df["prod"].isin(["UNK", "primary"]))
            & (df["head"].isin(["BinaryResponse"]))
            & (df["ent"] == "Primary")
        )
        _mask_28 = (
            (~df["prod"].isin(["UNK"]))
            & (df["head"].isin(["UNK"]))
            & (df["ent"] == "Primary")
        )
        _mask_29 = (
            (df["prod"].isin(["LIFESUP"]))
            & (df["head"].isin(["Inforce Amount"]))
            & (df["ent"] == "Primary")
            & (
                df["base"].isin(
                    ["Supplemental/Voluntary_Employee_Life_In_Force_Amount"]
                )
            )
        )
        _mask_31 = (
            (df["prod"].isin(["ADDSUP"]))
            & (df["head"].isin(["EffectiveDate"]))
            & (df["ent"] == "Primary")
            & (df["base"].isin(["Supplemental/Voluntary_Employee_AD&D_Effective_Date"]))
        )
        _mask_32 = (
            (df["prod"].isin(["UNK"]))
            & (df["head"].isin(["Zip"]))
            & (df["ent"] == "Primary")
            & (df["base"].isin(["Employee_Address_3"]))
        )
        _mask_34 = (
            (df["prod"].isin(["UNK"]))
            & (df["head"].isin(["UNK"]))
            & (df["ent"] == "Primary")
        )
        _mask_f1 = (df["ent"].isin(["Child"])) & df["prod"].isin(["CHLIFE", "CHADD"])

        df.loc[:, "prod"] = df["prod"].where(~(_mask_1), other="UNK")
        df.loc[:, "head"] = df["head"].where(~(_mask_8), other="AddressLine1")
        df.loc[:, "head"] = df["head"].where(~(_mask_9), other="HireDate")
        df.loc[:, "head"] = df["head"].where(~(_mask_10), other="BenefitClass")
        df.loc[:, "head"] = df["head"].where(~(_mask_13), other="Zip")
        df.loc[:, "head"] = df["head"].where(~(_mask_14), other="Grandfathered Amount")
        df.loc[:, "prod"] = df["prod"].where(~(_mask_17), other="LIFEVOL")
        df.loc[:, "prod"] = df["prod"].where(~(_mask_18), other="ADDVOL")
        df.loc[:, "prod"] = df["prod"].where(~(_mask_20), other="LIFESUP")
        df.loc[:, "ent"] = df["ent"].where(~(_mask_21), other="Spouse")
        df.loc[:, "ent"] = df["ent"].where(~(_mask_23), other="UNK")
        df.loc[:, "prod"] = df["prod"].where(~(_mask_24), other="LIFE")
        df.loc[:, "prod"] = df["prod"].where(~(_mask_25), other="LIFEVOL")
        df.loc[:, "head"] = df["head"].where(~(_mask_27), other="Applied For Amount")
        df.loc[:, "ent"] = df["ent"].where(~(_mask_28), other="UNK")
        df.loc[:, "prod"] = df["prod"].where(~(_mask_29), other="LIFEVOL")
        df.loc[:, "prod"] = df["prod"].where(~(_mask_31), other="ADDVOL")
        df.loc[:, "head"] = df["head"].where(~(_mask_32), other="State")
        df.loc[:, "ent"] = df["ent"].where(~(_mask_34), other="UNK")
        df.loc[:, "ent"] = df["ent"].where(~(_mask), other="Primary")
        df.loc[:, "ent"] = df["ent"].where(~(_mask_f1), other="Primary")

        return df

    def regex_filter_bank(row):
        pattern_dep = re.compile("^dependent_relationship_dependent.+(\d)")
        pattern_dep1 = re.compile("^dependent(-\d)?")
        pattern_child = re.compile("^.+(\d)")
        pattern_emp_add = re.compile("^employee_address(_\d)?")

        match_obj_emp_add = pattern_emp_add.match(row["base"].lower())

        if pattern_dep.match(row["base"].lower()):
            if (
                pattern_dep1.match(row["ent"].lower())
                and row["head"] == "Relationship"
                and row["prod"] == "UNK"
            ):
                row["ent"] = "Child-" + str(pattern_dep.match(row["base"].lower())[1])

        if row["ent"] == "Child" and row["prod"] == "UNK":
            if pattern_child.match(row["base"]):
                row["ent"] = "Child-" + str(pattern_child.match(row["base"].lower())[1])

        if match_obj_emp_add:
            if match_obj_emp_add[1]:
                if match_obj_emp_add[1][1] == "1":
                    row["head"] = "City"
                elif match_obj_emp_add[1][1] == "2":
                    row["head"] = "State"
                elif match_obj_emp_add[1][1] == "3":
                    row["head"] = "Zip"
            else:
                row["head"] = "AddressLine1"

        return row

    prod_dict = read_objs("prod_dict")
    head_dict = read_objs("head_dict")
    party_dict = read_objs("entity_dict")

    max_seq_len = 16
    pad = "pre"

    # load wordPiece tokenizer
    tokenizer = pkg_resources.resource_stream(
        "model_ffm", "data/p2_tokenizer_BERT_WP_v2.2.1.joblib"
    )
    bert_wp_loaded = joblib.load(tokenizer)

    # load trained model
    fp = pkg_resources.resource_filename(
        "model_ffm", "data/ffm_model_torch_v2.2.1.onnx"
    )
    loaded_model = onnxruntime.InferenceSession(fp)

    input_name = loaded_model.get_inputs()[0].name
    label_name1 = loaded_model.get_outputs()[0].name
    label_name2 = loaded_model.get_outputs()[1].name
    label_name3 = loaded_model.get_outputs()[2].name

    all_columns = kwargs.get("inputs").get("columns")

    token_ids_test = []
    processed_cols = []
    for column in all_columns:
        column = humanize(underscore(column))
        processed_cols.append(column)
        token_ids_test.append(bert_wp_loaded.encode(column).ids)

    # test_tokens = pad_sequences(
    #     token_ids_test,
    #     padding=pad,
    #     value=bert_wp_loaded.get_vocab()["[PAD]"],
    #     maxlen=max_seq_len,
    # )
    pred = []

    for A in range(len(token_ids_test)):
        t = max_seq_len - len(token_ids_test[A])
        if t >= 0:
            i = np.pad(token_ids_test[A], pad_width=(t, 0), mode="constant").astype(
                np.int32
            )
        else:
            i = np.array(token_ids_test[A][0:16]).astype(np.int32)
        i = i.reshape(1, -1)
        pred.append(
            loaded_model.run([label_name1, label_name2, label_name3], {input_name: i})
        )

    predictions_test = []
    predictions_test_0 = []
    predictions_test_1 = []
    predictions_test_2 = []

    for i in range(len(token_ids_test)):
        predictions_test_0.append(pred[i][0])
        predictions_test_1.append(pred[i][1])
        predictions_test_2.append(pred[i][2])

    predictions_test.append(predictions_test_0)
    predictions_test.append(predictions_test_1)
    predictions_test.append(predictions_test_2)

    r1_len = len(all_columns)
    logging.debug(f'{r1_len}')

    p1_test = np.argmax(np.array(predictions_test[0]).reshape(r1_len, -1), axis=1)
    p2_test = np.argmax(np.array(predictions_test[1]).reshape(r1_len, -1), axis=1)
    p3_test = np.argmax(np.array(predictions_test[2]).reshape(r1_len, -1), axis=1)

    prod_label = np.array([prod_dict[x] for x in p1_test]).reshape((-1, 1))
    header_label = np.array([head_dict[x] for x in p2_test]).reshape((-1, 1))
    entity_label = np.array([party_dict[x] for x in p3_test]).reshape((-1, 1))

    fp_1 = pkg_resources.resource_filename("model_ffm", "data/softmax.onnx")
    softmax_layer = onnxruntime.InferenceSession(fp_1)
    input_name_x = softmax_layer.get_inputs()[0].name
    label_name_x = softmax_layer.get_outputs()[0].name

    prob_1 = softmax_layer.run(
        [label_name_x],
        {input_name_x: np.array(predictions_test[0]).reshape(r1_len, -1)},
    )[0]
    prob_2 = softmax_layer.run(
        [label_name_x],
        {input_name_x: np.array(predictions_test[1]).reshape(r1_len, -1)},
    )[0]
    prob_3 = softmax_layer.run(
        [label_name_x],
        {input_name_x: np.array(predictions_test[2]).reshape(r1_len, -1)},
    )[0]

    prod_confidence = np.array(
        [round(float(x), 4) for x in list(np.max(prob_1, axis=1))]
    ).reshape((-1, 1))
    header_confidence = np.array(
        [round(float(x), 4) for x in list(np.max(prob_2, axis=1))]
    ).reshape((-1, 1))
    entity_confidence = np.array(
        [round(float(x), 4) for x in list(np.max(prob_3, axis=1))]
    ).reshape((-1, 1))

    pred_labels = np.hstack((prod_label, header_label, entity_label)).tolist()
    confidences = np.hstack(
        (prod_confidence, header_confidence, entity_confidence)
    ).tolist()

    # Apply new mapping (local_build:2.4.6)

    prediction_df = pd.DataFrame(pred_labels, columns=["prod", "head", "ent"])
    prediction_df["base"] = all_columns
    prediction_df = filter_bank(prediction_df)
    prediction_df = prediction_df.apply(regex_filter_bank, axis=1)
    new_labels = prediction_df.values.tolist()
    res = [
        [(a, b) for a, b in zip(ll, cl)]
        for ll, cl in zip(new_labels, confidences)  # noqa
    ]

    pred_list = [{entity: prediction} for entity, prediction in zip(all_columns, res)]
    pattern = re.compile("^blank_header_\d+$")
    unknown_triplet = [("UNK", 1.0), ("UNK", 1.0), ("UNK", 1.0)]
    prediction_list = [
        {header: unknown_triplet if pattern.match(header) else pred}
        for item in pred_list
        for header, pred in item.items()
    ]
    # print(f"{pred_labels=}")
    logging.debug(f'{pred_labels}')
    return [
        {
            "inputDataSource": f"{dataset_id}:0",
            "entityId": f"{dataset_id}",
            # "entityId": f"{dataset_id}:{list(pr.keys())[0]}",
            "predictedResult": prediction_list,
            # "predictedResult": pr,
        }
        # for pr in prediction_list
    ]


# if __name__ == "__main__":
#     columns = [
#         "Dependent CHILD #3 SSN",
#         "Child#1 DOB",
#         "Child 2 DOB",
#         "Ch1.LastName",
#         "ACC Effective Date",
#         "Member_Information_Employee_Benefit_Class",
#         "Employee_Address_1",
#         "Member_Information_Last_Name",
#         "blank_header_1",
#         "blank_header_20",
#     ]
#     # columns = []
#     results = predict(
#         model_name="model_ffm",
#         artifacts=["data/bert_wp_tok_updated_v2.joblib"],
#         model_path="data/FFM_new_prod_labels_v2.h5",
#         inputs={"datasetId": "spr:dataset_id", "columns": columns},
#     )
#     print(f"{results=}")
