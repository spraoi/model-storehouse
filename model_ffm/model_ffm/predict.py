def predict(**kwargs):
    """Generate predictions for input data

    Returns:
        List[Dict]: A list of dictionaries representing the prediction data.
              Each dictionary contains the following keys:
              - inputDataSource (string): Dataset Id
              - entityId (string): Entity Id
              - predictedResult (list): A list of prediction results
    """
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
        # resource_loc = f"data/{keyword}{ARTIFACT_VERSION}.{method}" # For local debug
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

        (
            loaded_model,
            input_name,
            [label_name1, label_name2, label_name3],
        ) = load_resources("ffm_model_torch_v", "onnx")
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
        softmax_layer, input_name_x, [label_name_x] = load_resources(
            "softmax_v", "onnx"
        )
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

    def filter_bank_numpy_version(prediction_arr):
        """Apply DBN specific prediction remapping on all predictions.

        Args:
            prediction_arr (numpy.nd_array): A numpy multidimensional array containing predictions and input field

        Returns:
            numpy.nd_array: A numpy multidimensional array containing remapped predictions and original input field
        """
        mask = (np.isin(prediction_arr[:, 3], ["UNK"])) & ~(
            np.isin(prediction_arr[:, 2], ["GroupNumber", "AccountNumber"])
        )
        mask_1 = (np.isin(prediction_arr[:, 3], (["Spouse", "Child", "Dependent"]))) & (
            np.isin(prediction_arr[:, 1], (["VIS", "DEN"]))
        )
        mask_9 = np.isin(prediction_arr[:, 0], (["Eligibility_DOH"]))
        mask_8 = (np.isin(prediction_arr[:, 0], (["Employee_Address"]))) & (
            np.isin(prediction_arr[:, 2], ["Zip"])
        )
        mask_10 = np.isin(prediction_arr[:, 0], (["Class/Set"]))
        mask_13 = (np.isin(prediction_arr[:, 0], (["Employee_Address_3"]))) & (
            np.isin(prediction_arr[:, 2], ["AddressLine3"])
        )
        mask_14 = np.isin(prediction_arr[:, 2], ["GF_BenefitAmount"])
        mask_17 = (
            np.isin(
                prediction_arr[:, 0],
                [
                    "Supplemental/Voluntary_Employee_Life_In_Force_Amount",
                    "Supplemental/Voluntary_Employee_Life_Applied_For_Amount",
                ],
            )
        ) & (np.isin(prediction_arr[:, 1], ["LIFESUP"]))
        mask_18 = (
            np.isin(
                prediction_arr[:, 0],
                [
                    "Supplemental/Voluntary_Employee_AD&D_Applied_For_Amount",
                    "Supplemental/Voluntary_Employee_AD&D_In_Force_Amount",
                ],
            )
        ) & (np.isin(prediction_arr[:, 1], ["ADDSUP"]))
        mask_20 = (
            np.isin(
                prediction_arr[:, 0],
                [
                    "EMPLOYEE_Voluntary/Supplemental_Coverage_LIFE_Benefit_Amount",
                    "EMPLOYEE_Voluntary/SupplementalCoverage_LIFE_Benefit_Amount",
                ],
            )
        ) & (np.isin(prediction_arr[:, 1], ["LIFEVOL"]))
        mask_21 = (
            np.isin(
                prediction_arr[:, 0], ["Dependent_Relationship_Dependent_Relationship"]
            )
        ) & (np.isin(prediction_arr[:, 3], ["Dependent"]))
        mask_23 = (
            (np.isin(prediction_arr[:, 2], ["AccountNumber"]))
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
            & (np.isin(prediction_arr[:, 1], ["UNK"]))
        )
        mask_24 = (
            (
                np.isin(
                    prediction_arr[:, 0],
                    ["Basic_Life_In_Force_Amount", "Basic_Life_Applied_For_Amount"],
                )
            )
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
            & (np.isin(prediction_arr[:, 1], ["LTD"]))
        )
        mask_25 = (
            (
                np.isin(
                    prediction_arr[:, 0],
                    ["Supplemental/Voluntary_Employee_Life_Effective_Date"],
                )
            )
            & (np.isin(prediction_arr[:, 2], ["EffectiveDate"]))
            & (np.isin(prediction_arr[:, 1], ["LIFESUP"]))
        )
        mask_27 = (
            (~(np.isin(prediction_arr[:, 1], ["UNK", "primary"])))
            & (np.isin(prediction_arr[:, 2], ["BinaryResponse"]))
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
        )
        mask_28 = (
            (~(np.isin(prediction_arr[:, 1], ["UNK"])))
            & (np.isin(prediction_arr[:, 2], ["UNK"]))
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
        )
        mask_29 = (
            ((np.isin(prediction_arr[:, 1], ["LIFESUP"])))
            & (np.isin(prediction_arr[:, 2], ["Inforce Amount"]))
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
            & (
                np.isin(
                    prediction_arr[:, 0],
                    ["Supplemental/Voluntary_Employee_Life_In_Force_Amount"],
                )
            )
        )
        mask_31 = (
            ((np.isin(prediction_arr[:, 1], ["ADDSUP"])))
            & (np.isin(prediction_arr[:, 2], ["EffectiveDate"]))
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
            & (
                np.isin(
                    prediction_arr[:, 0],
                    ["Supplemental/Voluntary_Employee_AD&D_Effective_Date"],
                )
            )
        )
        mask_32 = (
            ((np.isin(prediction_arr[:, 1], ["UNK"])))
            & (np.isin(prediction_arr[:, 2], ["Zip"]))
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
            & (np.isin(prediction_arr[:, 0], ["Employee_Address_3"]))
        )
        mask_34 = (
            ((np.isin(prediction_arr[:, 1], ["UNK"])))
            & (np.isin(prediction_arr[:, 2], ["UNK"]))
            & (np.isin(prediction_arr[:, 3], ["Primary"]))
        )
        mask_35 = (~(np.isin(prediction_arr[:, 1], ["UNK"]))) & (
            np.isin(
                prediction_arr[:, 2],
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
                ],
            )
        )
        mask_f1 = (np.isin(prediction_arr[:, 3], ["Child"])) & (
            np.isin(prediction_arr[:, 1], ["CHLIFE", "CHADD"])
        )

        prediction_arr[mask_1][:, 1] = "UNK"
        prediction_arr[mask_8][:, 2] = "AddressLine1"
        prediction_arr[mask_9][:, 2] = "HireDate"
        prediction_arr[mask_10][:, 2] = "BenefitClass"
        prediction_arr[mask_13][:, 2] = "Zip"
        prediction_arr[mask_14][:, 2] = "Grandfathered Amount"
        prediction_arr[mask_17][:, 1] = "LIFEVOL"
        prediction_arr[mask_18][:, 1] = "ADDVOL"
        prediction_arr[mask_20][:, 1] = "LIFESUP"
        prediction_arr[mask_21][:, 3] = "Spouse"
        prediction_arr[mask_23][:, 3] = "UNK"
        prediction_arr[mask_24][:, 1] = "LIFE"
        prediction_arr[mask_25][:, 1] = "LIFEVOL"
        prediction_arr[mask_27][:, 2] = "Applied For Amount"
        prediction_arr[mask_28][:, 3] = "UNK"
        prediction_arr[mask_29][:, 1] = "LIFEVOL"
        prediction_arr[mask_31][:, 1] = "ADDVOL"
        prediction_arr[mask_32][:, 2] = "State"
        prediction_arr[mask_34][:, 3] = "UNK"
        prediction_arr[mask][:, 3] = "Primary"
        prediction_arr[mask_f1][:, 3] = "Primary"
        prediction_arr[mask_35][:, 1] = "UNK"

        return prediction_arr

    def regex_filter_bank_numpy_version(row):
        """Remaps prediction using RegEx filters on each data instance(row).

        Args:
            row (numpy.array): A one dimensional numpy array containing each prediction instance

        Returns:
            numpy.array: A one dimensional numpy array containing modified prediction instance
        """
        match_obj_emp_add = PATTERN_EMP_ADD.match(row[0].lower())

        if PATTERN_DEP.match(row[0].lower()):
            if (
                PATTERN_DEP1.match(row[3].lower())
                and row[2] == "Relationship"
                and row[1] == "UNK"
            ):
                row[3] = "Child-" + PATTERN_DEP.match(row[1].lower())[1]

        if row[3] == "Child" and row[1] == "UNK":
            if match_obj := PATTERN_CHILD.match(row[1].lower()):
                row[3] = "Child-" + match_obj[1]

        if match_obj_emp_add:
            if match_obj_emp_add[1]:
                if match_obj_emp_add[1][1] == "1":
                    row[2] = "City"
                elif match_obj_emp_add[1][1] == "2":
                    row[2] = "State"
                elif match_obj_emp_add[1][1] == "3":
                    row[2] = "Zip"
            else:
                row[2] = "AddressLine1"

        return row

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
    unpacked_confidences = [(list(axes)) for axes in list(zip(*predictions))]
    resolved_confidences = map_confidences(unpacked_confidences)

    prediction_array = np.array(
        list(
            [item] + sub_list
            for item, sub_list in zip(all_columns, resolved_predicted_labels)
        )
    )

    # Apply post prediction mapping (local_build:3.0.0)
    prediction_array_s1 = filter_bank_numpy_version(prediction_array)
    prediction_array_s2 = np.apply_along_axis(
        regex_filter_bank_numpy_version, axis=1, arr=prediction_array_s1
    )
    prediction_array_final = [
        sublist[1:] + [sublist[0]] for sublist in prediction_array_s2.tolist()
    ]

    resolved_results = [
        [(a, b) for a, b in zip(ll, cl)]
        for ll, cl in zip(prediction_array_final, resolved_confidences)
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
