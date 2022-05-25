def predict(**kwargs):

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from inflection import humanize, underscore
    import joblib
    import pkg_resources
    import re

    dataset_id = kwargs.get("inputs").get("datasetId")

    if not kwargs.get("inputs").get("columns"):
        return [
            {
                "inputDataSource": f"{dataset_id}:0",
                "entityId": dataset_id,
                "predictedResult": [],
            }
        ]

    # Static Mappings for the Model
    model_config = {
        "MAX_SEQ_LEN": 16,
        "PADDING": "pre",
        "label_mapping": {
            "entity_map": {
                "Primary": 26,
                "Spouse": 27,
                "UNK": 28,
                "Child": 10,
                "Dependent": 18,
                "Beneficiary-1": 0,
                "Beneficiary-2": 1,
                "Beneficiary-3": 2,
                "Employer": 25,
                "CB-1": 5,
                "CB-2": 6,
                "Dependent-1": 19,
                "Dependent-2": 20,
                "Dependent-3": 21,
                "Dependent-4": 22,
                "Dependent-5": 23,
                "Dependent-6": 24,
                "Child-1": 11,
                "Child-2": 12,
                "Child-3": 13,
                "Child-4": 14,
                "Child-5": 15,
                "Child-6": 16,
                "Beneficiary-4": 3,
                "Beneficiary-5": 4,
                "CB-3": 7,
                "CB-4": 8,
                "CB-5": 9,
                "Child-7": 17,
            },
            "product_map": {
                "UNK": 30,
                "LIFE": 19,
                "ACC": 0,
                "ADD": 1,
                "ASOFEE": 5,
                "STD": 28,
                "HEALTH": 18,
                "DEN": 14,
                "CRIT": 13,
                "CIW": 9,
                "DEPLIFE": 15,
                "ADD LIFE": 2,
                "LTD": 22,
                "CHADD": 6,
                "CHLIFE": 8,
                "CHCRIT": 7,
                "COBRA": 10,
                "VIS": 31,
                "COBRAVIS": 12,
                "SPCRIT": 25,
                "COBRADEN": 11,
                "EAPFEE": 16,
                "LIFEVOL": 21,
                "ADDVOL": 4,
                "ADDSUP": 3,
                "LIFESUP": 20,
                "SPADD": 24,
                "SPLIFE": 26,
                "STDVOL": 29,
                "LTDVOL": 23,
                "FMLA": 17,
                "STADD": 27,
            },
            "header_map": {
                "NumWorkingHoursWeek": 55,
                "DateOfBirth": 21,
                "FirstName": 31,
                "Gender": 36,
                "LastName": 48,
                "Salary": 67,
                "Age": 7,
                "Premium": 58,
                "GenericInd": 38,
                "EligibilityInd": 27,
                "EffectiveDate": 25,
                "TerminationDate": 74,
                "UNK": 78,
                "CoverageTier": 19,
                "AccountNumber": 0,
                "AppliedForBenefitAmount": 10,
                "Address": 2,
                "AddressLine1": 3,
                "BinaryResponse": 15,
                "Carrier": 16,
                "Action": 1,
                "PlanCode": 57,
                "EmploymentStatus": 28,
                "GenericDate": 37,
                "Zip": 83,
                "AddressLine2": 4,
                "Country": 18,
                "AddressLine3": 5,
                "AdjustDeductAmt": 6,
                "BenefitClass": 12,
                "CurrencySalary": 20,
                "FullName": 32,
                "BillingDivision": 14,
                "PhoneNumber": 56,
                "GroupNumber": 40,
                "BenefitPercentage": 13,
                "SSN": 66,
                "MiddleInitial": 51,
                "Relationship": 65,
                "BeneficiaryType": 11,
                "Product": 61,
                "TimeFreq": 76,
                "EligGroup": 26,
                "JobTitle": 47,
                "NumDependents": 54,
                "EOI_Amount": 24,
                "GF_Indicator": 34,
                "GuaranteeIssueInd": 41,
                "GF_BenefitAmount": 33,
                "PremiumFreq": 59,
                "Provider": 62,
                "WaiveReason": 81,
                "City": 17,
                "TerminationReasonCode": 75,
                "NotesOrDesc": 53,
                "USCounty": 79,
                "HireDate": 42,
                "DriversLicense": 23,
                "MiddleName": 52,
                "State": 72,
                "DisabilityInd": 22,
                "TobaccoUserOrSmokerInd": 77,
                "IDType": 43,
                "SeqNumber": 71,
                "emailAddress": 84,
                "SalaryFreq": 69,
                "InforceInd": 46,
                "TaxStatus": 73,
                "Reason": 63,
                "MaritalStatus": 49,
                "SalaryEffectiveDate": 68,
                "RehireDate": 64,
                "Units": 80,
                "WorkLocation": 82,
                "AgeGroup": 8,
                "FLSAStatus": 29,
                "FSAamount": 30,
                "Alt_IdentityNumber": 9,
                "Generic_ID": 39,
                "MemberID": 50,
                "IdentityNumber": 44,
                "SecondaryAccountNumber": 70,
                "InforceAmount": 45,
                "PriorCarrierInd": 60,
                "GI_Amount": 35,
            },
        },
    }

    max_seq_len = model_config.get("MAX_SEQ_LEN")
    pad = model_config.get("PADDING")

    # load wordPiece tokenizer
    tokenizer = pkg_resources.resource_stream(
        "model_ffm", "data/bert_wp_tok_updated_v2.joblib"
    )
    bert_wp_loaded = joblib.load(tokenizer)

    # load trained model
    fp = pkg_resources.resource_filename("model_ffm", "data/FFM_new_prod_labels_v2.h5")
    loaded_model = tf.keras.models.load_model(fp)

    # load categories:index mappings
    inv_prod_dict = model_config.get("label_mapping").get("product_map")
    prod_dict = {v: k for k, v in inv_prod_dict.items()}

    inv_head_dict = model_config.get("label_mapping").get("header_map")
    head_dict = {v: k for k, v in inv_head_dict.items()}

    inv_party_dict = model_config.get("label_mapping").get("entity_map")
    party_dict = {v: k for k, v in inv_party_dict.items()}

    all_columns = kwargs.get("inputs").get("columns")

    token_ids_test = []
    processed_cols = []
    for column in all_columns:
        column = humanize(underscore(column))
        processed_cols.append(column)
        token_ids_test.append(bert_wp_loaded.encode(column).ids)

    test_tokens = pad_sequences(
        token_ids_test,
        padding=pad,
        value=bert_wp_loaded.get_vocab()["[PAD]"],
        maxlen=max_seq_len,
    )

    predictions_test = loaded_model.predict(test_tokens)

    p1_test = np.argmax(predictions_test[0], axis=1)
    p2_test = np.argmax(predictions_test[1], axis=1)
    p3_test = np.argmax(predictions_test[2], axis=1)

    prod_label = np.array([prod_dict[x] for x in p1_test]).reshape((-1, 1))
    header_label = np.array([head_dict[x] for x in p2_test]).reshape((-1, 1))
    entity_label = np.array([party_dict[x] for x in p3_test]).reshape((-1, 1))

    prob_1 = tf.nn.softmax(predictions_test[0], axis=1)
    prob_2 = tf.nn.softmax(predictions_test[1], axis=1)
    prob_3 = tf.nn.softmax(predictions_test[2], axis=1)

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
    # print(f"{pred_labels=}")

    # Apply new rule for entity 04/22/2022
    # if predicted entity is UNK and predicted label is NOT GroupNumber or AccountNumber
    # then set predicted entity to Primary
    new_labels = [
        [
            p[0],
            p[1],
            "Primary"
            if p[2] == "UNK" and p[1] not in ["GroupNumber", "AccountNumber"]
            else p[2],
        ]
        for p in pred_labels  # noqa
    ]
    res = [
        [(a, b) for a, b in zip(ll, cl)]
        for ll, cl in zip(new_labels, confidences)  # noqa
    ]
    # print(f"{res=}")

    pred_list = [{entity: prediction} for entity, prediction in zip(all_columns, res)]
    pattern = re.compile("^blank_header_\d+$")
    unknown_triplet = [("UNK", 1.0), ("UNK", 1.0), ("UNK", 1.0)]
    prediction_list = [
        {header: unknown_triplet if pattern.match(header) else pred}
        for item in pred_list
        for header, pred in item.items()
    ]
    # print(f"{prediction_list=}")

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


if __name__ == "__main__":
    columns = [
        "Dependent CHILD #3 SSN",
        "Child#1 DOB",
        "Child 2 DOB",
        "Ch1.LastName",
        "ACC Effective Date",
        "Member_Information_Employee_Benefit_Class",
        "Member_Information_Last_Name",
        "blank_header_1",
        "blank_header_20",
    ]
    # columns = []
    results = predict(
        model_name="model_ffm",
        artifacts=["data/bert_wp_tok_updated_v2.joblib"],
        model_path="data/FFM_new_prod_labels_v2.h5",
        inputs={"datasetId": "spr:dataset_id", "columns": columns},
    )
    print(f"{results=}")
