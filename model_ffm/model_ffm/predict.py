def predict(**kwargs):
    """Generate predictions for input data

    Returns:
        List[Dict]: A list of dictionaries representing the prediction data.
              Each dictionary contains the following keys:
              - inputDataSource (string): Dataset Id
              - entityId (string): Entity Id
              - predictedResult (list): A list of prediction results
    """
    import pandas as pd
    import numpy as np
    from inflection import humanize, underscore
    import pkg_resources
    import re
    import onnxruntime
    import joblib

    ARTIFACT_VERSION = "3.0.0"
    MAX_SEQ_LENGTH = 16  # Padding threshold
    # Regex patterns for post prediction logic
    dep_relation_pattern = r"^dependent_relationship_dependent.+(\d)"
    dep_pattern = r"^dependent(-\d)?"
    child_pattern = r"^.+(\d)"
    emp_address_pattern = r"^employee_address(_\d)?"
    blank_pattern = r"^blank_header_\d+$"
    PATTERN_BLANK = re.compile(blank_pattern)
    PATTERN_DEP = re.compile(dep_relation_pattern)
    PATTERN_DEP1 = re.compile(dep_pattern)
    PATTERN_CHILD = re.compile(child_pattern)
    PATTERN_EMP_ADD = re.compile(emp_address_pattern)

    dataset_id = kwargs.get("inputs").get("datasetId")
    if not kwargs.get("inputs").get("columns"):
        return [
            {
                "inputDataSource": f"{dataset_id}:0",
                "entityId": dataset_id,
                "predictedResult": [],
            }
        ]

    def filter_bank(df):
        """Apply DBN specific prediction remapping on entire dataframe

        Args:
            df (pandas.DataFrame): A dataframe containing predictions

        Returns:
            pandas.DataFrame: A dataframe containing remapped predictions
        """
        mask = (df["ent"].isin(["UNK"])) & ~df["head"].isin(
            ["GroupNumber", "AccountNumber"]
        )
        mask_1 = (df["ent"].isin(["Spouse", "Child", "Dependent"])) & (
            df["prod"].isin(["VIS", "DEN"])
        )
        mask_9 = df["base"].isin(["Eligibility_DOH"])
        mask_8 = (df["base"].isin(["Employee_Address"])) & (df["head"] == "Zip")
        mask_10 = df["base"].isin(["Class/Set"])
        mask_13 = (df["base"].isin(["Employee_Address_3"])) & (
                df["head"] == "AddressLine3"
        )
        mask_14 = df["head"] == "GF_BenefitAmount"
        mask_17 = (
                      df["base"].isin(
                          [
                              "Supplemental/Voluntary_Employee_Life_In_Force_Amount",
                              "Supplemental/Voluntary_Employee_Life_Applied_For_Amount",
                          ]
                      )
                  ) & (df["prod"] == "LIFESUP")
        mask_18 = (
                      df["base"].isin(
                          [
                              "Supplemental/Voluntary_Employee_AD&D_Applied_For_Amount",
                              "Supplemental/Voluntary_Employee_AD&D_In_Force_Amount",
                          ]
                      )
                  ) & (df["prod"] == "ADDSUP")
        mask_20 = (
                      df["base"].isin(
                          [
                              "EMPLOYEE_Voluntary/Supplemental_Coverage_LIFE_Benefit_Amount",
                              "EMPLOYEE_Voluntary/SupplementalCoverage_LIFE_Benefit_Amount",
                          ]
                      )
                  ) & (df["prod"] == "LIFEVOL")
        mask_21 = (
                      df["base"].isin(["Dependent_Relationship_Dependent_Relationship"])
                  ) & (df["ent"] == "Dependent")
        mask_23 = (
                (df["head"].isin(["AccountNumber"]))
                & (df["ent"] == "Primary")
                & (df["prod"] == "UNK")
        )
        mask_24 = (
                (
                    df["base"].isin(
                        ["Basic_Life_In_Force_Amount", "Basic_Life_Applied_For_Amount"]
                    )
                )
                & (df["ent"] == "Primary")
                & (df["prod"] == "LTD")
        )
        mask_25 = (
                (df["base"].isin(["Supplemental/Voluntary_Employee_Life_Effective_Date"]))
                & (df["head"].isin(["EffectiveDate"]))
                & (df["prod"] == "LIFESUP")
        )
        mask_27 = (
                (~df["prod"].isin(["UNK", "primary"]))
                & (df["head"].isin(["BinaryResponse"]))
                & (df["ent"] == "Primary")
        )
        mask_28 = (
                (~df["prod"].isin(["UNK"]))
                & (df["head"].isin(["UNK"]))
                & (df["ent"] == "Primary")
        )
        mask_29 = (
                (df["prod"].isin(["LIFESUP"]))
                & (df["head"].isin(["Inforce Amount"]))
                & (df["ent"] == "Primary")
                & (
                    df["base"].isin(
                        ["Supplemental/Voluntary_Employee_Life_In_Force_Amount"]
                    )
                )
        )
        mask_31 = (
                (df["prod"].isin(["ADDSUP"]))
                & (df["head"].isin(["EffectiveDate"]))
                & (df["ent"] == "Primary")
                & (df["base"].isin(["Supplemental/Voluntary_Employee_AD&D_Effective_Date"]))
        )
        mask_32 = (
                (df["prod"].isin(["UNK"]))
                & (df["head"].isin(["Zip"]))
                & (df["ent"] == "Primary")
                & (df["base"].isin(["Employee_Address_3"]))
        )
        mask_34 = (
                (df["prod"].isin(["UNK"]))
                & (df["head"].isin(["UNK"]))
                & (df["ent"] == "Primary")
        )
        mask_35 = (~df["prod"].isin(["UNK"])) & (
            df["head"].isin(
                [
                    "DateOfBirth",
                    "FirstName",
                    "Gender",
                    "LastName",
                    "Address",
                    "AddressLine2",
                    "City",
                    "State",
                    "AddressLine1",
                    "Zip",
                    "Relationship",
                ]
            )
        )
        mask_f1 = (df["ent"].isin(["Child"])) & df["prod"].isin(["CHLIFE", "CHADD"])

        df.loc[:, "prod"] = df["prod"].where(~(mask_1), other="UNK")
        df.loc[:, "head"] = df["head"].where(~(mask_8), other="AddressLine1")
        df.loc[:, "head"] = df["head"].where(~(mask_9), other="HireDate")
        df.loc[:, "head"] = df["head"].where(~(mask_10), other="BenefitClass")
        df.loc[:, "head"] = df["head"].where(~(mask_13), other="Zip")
        df.loc[:, "head"] = df["head"].where(~(mask_14), other="Grandfathered Amount")
        df.loc[:, "prod"] = df["prod"].where(~(mask_17), other="LIFEVOL")
        df.loc[:, "prod"] = df["prod"].where(~(mask_18), other="ADDVOL")
        df.loc[:, "prod"] = df["prod"].where(~(mask_20), other="LIFESUP")
        df.loc[:, "ent"] = df["ent"].where(~(mask_21), other="Spouse")
        df.loc[:, "ent"] = df["ent"].where(~(mask_23), other="UNK")
        df.loc[:, "prod"] = df["prod"].where(~(mask_24), other="LIFE")
        df.loc[:, "prod"] = df["prod"].where(~(mask_25), other="LIFEVOL")
        df.loc[:, "head"] = df["head"].where(~(mask_27), other="Applied For Amount")
        df.loc[:, "ent"] = df["ent"].where(~(mask_28), other="UNK")
        df.loc[:, "prod"] = df["prod"].where(~(mask_29), other="LIFEVOL")
        df.loc[:, "prod"] = df["prod"].where(~(mask_31), other="ADDVOL")
        df.loc[:, "head"] = df["head"].where(~(mask_32), other="State")
        df.loc[:, "ent"] = df["ent"].where(~(mask_34), other="UNK")
        df.loc[:, "ent"] = df["ent"].where(~(mask), other="Primary")
        df.loc[:, "ent"] = df["ent"].where(~(mask_f1), other="Primary")
        df.loc[:, "prod"] = df["prod"].where(~(mask_35), other="UNK")
        return df

    def regex_filter_bank(row):
        """Remaps prediction using RegEx filters on each data instance(row).

        Args:
            row (pandas.Series): A pandas series containing each prediction instance

        Returns:
            pandas.Series: A pandas series containing modified prediction instance
        """
        match_obj_emp_add = PATTERN_EMP_ADD.match(row["base"].lower())

        if PATTERN_DEP.match(row["base"].lower()):
            if (
                    PATTERN_DEP1.match(row["ent"].lower())
                    and row["head"] == "Relationship"
                    and row["prod"] == "UNK"
            ):
                row["ent"] = "Child-" + PATTERN_DEP.match(row["base"].lower())[1]

        if row["ent"] == "Child" and row["prod"] == "UNK":
            if match_obj := PATTERN_CHILD.match(row["base"].lower()):
                row["ent"] = "Child-" + match_obj[1]

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

    def load_resources(keyword, method):
        """Load pickle, joblib or onnx serialized objects

        Args:
            keyword (string): Filename
            method (string): File extension

        Returns:
            Any: The return value is based on the condition.
              If the object is loaded using onnx, the loaded model along with its input dimensions and a list of output dimensions are returned.

              If the object is loaded using pickle or joblib, the loaded object is returned.
        """
        # resource_loc = f"data/{keyword}{ARTIFACT_VERSION}.{method}"  # For local debug
        resource_loc = pkg_resources.resource_stream(
            "model_ffm", f"data/{keyword}{ARTIFACT_VERSION}.{method}"
        )
        if method == "onnx":
            loaded_obj = onnxruntime.InferenceSession(resource_loc)
            input_dim = loaded_obj.get_inputs()[0].name
            output_dim = [
                loaded_obj.get_outputs()[dimension].name
                for dimension in range(0, len(loaded_obj.get_outputs()))
            ]
            return loaded_obj, input_dim, output_dim
        else:
            loaded_obj = joblib.load(resource_loc)
            return loaded_obj

    # Pre-pad and prune sequences with 16 characters as a threshold
    def dynamic_pad_and_predict(input_tokens, max_seq_length):
        """Dynamically pad sequences and run predictions on them

        Args:
            input_tokens (List): A list of tokens to pad and predict
            max_seq_length (Int): Maximum sequence length limit

        Returns:
            List: A list of predictions
        """

        loaded_model,input_name,[label_name1, label_name2, label_name3] = load_resources("ffm_model_torch_v", "onnx")
        predictions = []
        for token in input_tokens:
            dynamic_pad = max_seq_length - len(token)
            if dynamic_pad >= 0:
                padded_seq = np.pad(
                    token, pad_width=(dynamic_pad, 0), mode="constant"
                ).astype(np.int32)
            else:
                padded_seq = np.array(token[0:16]).astype(np.int32)
            padded_seq = padded_seq.reshape(1, -1)
            # Run predictions
            predictions.append(
                loaded_model.run(
                    [label_name1, label_name2, label_name3], {input_name: padded_seq}
                )
            )
        return predictions

    def map_predictions_helper_max(prediction_list):
        """Get the prediction with maximum confidence across axes. Dimension[0] equals to the size of input data

        Args:
            prediction_list (List): A list containing one of the three prediction targets

        Returns:
            np.array: Returns a numpy array containing the predicted class
        """
        reshaped_prediction = np.array(prediction_list).reshape(run_len, -1)
        return np.argmax(reshaped_prediction, axis=1)

    def map_confidences_helper_max(confidence_list):
        """Generate confidence scores for each of the three targets independantly rounded of to 4 decimal places.

        Args:
            confidence_list (List): A list containing confidences of one of the three prediction targets

        Returns:
            np.array: Returns a numpy array containing confidences
        """
        return np.array(
            [round(float(x), 4) for x in list(np.max(confidence_list, axis=1))]
        ).reshape((-1, 1))

    def map_predictions(predictions):
        """Map prediction classes to their corresponding labels from training time. Uses serialized dictionaries from training time.

        Args:
            predictions (List[List, List, List]): A list of lists containing predictions of all three targets

        Returns:
            List: Returns a List of predictions mapped to their target labels.
        """
        product_label = np.array([prod_dict[x] for x in predictions[0]]).reshape(
            (-1, 1)
        )
        header_label = np.array([head_dict[x] for x in predictions[1]]).reshape((-1, 1))
        entity_label = np.array([party_dict[x] for x in predictions[2]]).reshape(
            (-1, 1)
        )

        predicted_labels = np.hstack(
            (product_label, header_label, entity_label)
        ).tolist()
        return predicted_labels

    def map_confidences(confidences):
        """Generate prediction confidences for each of the target axes.

        Args:
            confidences (List[List, List, List]): A list of lists containing predictions of all three targets

        Returns:
            List: Returns a list containing prediction confidences
        """
        softmax_layer, input_name_x, [label_name_x] = load_resources("softmax_v", "onnx")
        probabilities = []
        for dim in range(len(confidences)):
            probabilities.append(
                softmax_layer.run(
                    [label_name_x],
                    {input_name_x: np.array(confidences[dim]).reshape(run_len, -1)},
                )[0]
            )
        probabilities = [map_confidences_helper_max(axes) for axes in probabilities]
        return np.hstack(
            (probabilities[0], probabilities[1], probabilities[2])
        ).tolist()

    prod_dict = load_resources("prod_dict", "pkl")
    head_dict = load_resources("head_dict", "pkl")
    party_dict = load_resources("entity_dict", "pkl")
    bert_wp_loaded = load_resources("tokenzier_v", "joblib")

    all_columns = kwargs.get("inputs").get("columns")

    tokens = []
    processed_cols = []
    # Preprocess and tokenize the character sequence
    for column in all_columns:
        column = humanize(underscore(column))
        processed_cols.append(column)
        tokens.append(bert_wp_loaded.encode(column).ids)

    run_len = len(all_columns)

    predictions = dynamic_pad_and_predict(tokens, MAX_SEQ_LENGTH)

    unpacked_predictions = [
        map_predictions_helper_max(list(axes)) for axes in list(zip(*predictions))
    ]
    resolved_predicted_labels = map_predictions(unpacked_predictions)
    unpacked_confidences = [
        (list(axes)) for axes in list(zip(*predictions))
    ]
    resolved_confidences = map_confidences(unpacked_confidences)

    # Apply post prediction mapping (local_build:3.0.0)
    prediction_df = pd.DataFrame(
        resolved_predicted_labels, columns=["prod", "head", "ent"]
    )
    prediction_df["base"] = all_columns
    prediction_df = filter_bank(prediction_df)
    prediction_df = prediction_df.apply(regex_filter_bank, axis=1)
    new_labels = prediction_df.values.tolist()

    resolved_results = [
        [(a, b) for a, b in zip(ll, cl)]
        for ll, cl in zip(new_labels, resolved_confidences)  # noqa
    ]

    pred_list = [
        {entity: prediction}
        for entity, prediction in zip(all_columns, resolved_results)
    ]

    unknown_triplet = [("UNK", 1.0), ("UNK", 1.0), ("UNK", 1.0)]
    prediction_list = [
        {header: unknown_triplet if PATTERN_BLANK.match(header) else pred}
        for item in pred_list
        for header, pred in item.items()
    ]

    return [
        {
            "inputDataSource": f"{dataset_id}:0",
            "entityId": f"{dataset_id}",
            "predictedResult": prediction_list,
        }
    ]

# if __name__ == "__main__":
#     columns = [
#         "Dependent CHILD #3 SSN",
#         "Member_Information_Last_Name",
#         "Child_information_(1_age",
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
#     results = predict(
#         model_name="model_ffm",
#         artifacts=["data/bert_wp_tok_updated_v2.joblib"],
#         model_path="data/FFM_new_prod_labels_v2.h5",
#         inputs={"datasetId": "spr:dataset_id", "columns": columns},
#     )
#     print(f"{results=}")
