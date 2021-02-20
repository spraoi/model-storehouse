def predict(**kwargs):

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from inflection import humanize, underscore
    import joblib
    import pkg_resources

    # Static Mappings for the Model
    model_config = {
        "MAX_SEQ_LEN": 16,
        "PADDING": "pre",
        "label_mapping": {
            "entity_map": {
                "Primary": 26,
                "unk": 28,
                "Spouse": 27,
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
                "unk": 29,
                "LIFE": 15,
                "ACCIB": 0,
                "ADD": 2,
                "ASOFEE": 6,
                "STDATP": 23,
                "HEALTH": 14,
                "Dental": 11,
                "CI": 7,
                "CIW": 9,
                "ADD LIFE": 3,
                "LTD": 18,
                "STD": 21,
                "STDCORE": 25,
                "ADDSUP1": 4,
                "GF": 13,
                "LIFSUP1": 17,
                "LIFEVOL": 16,
                "ADDVOL": 5,
                "CIV": 8,
                "COBRA": 10,
                "VISB": 27,
                "EAPFEE": 12,
                "STDVOL": 26,
                "LTDVOL": 20,
                "STDBU": 24,
                "VISV": 28,
                "ACCIV": 1,
                "STDASO ": 22,
                "LTDBU": 19,
            },
            "header_map": {
                "NumWorkingHoursWeek": 49,
                "DateOfBirth": 22,
                "FirstName": 31,
                "Gender": 33,
                "LastName": 43,
                "Salary": 61,
                "AgeYears": 9,
                "Premium": 53,
                "GenericInd": 35,
                "EligibilityInd": 27,
                "EffectiveDate": 25,
                "TerminationDate": 66,
                "unk": 78,
                "CoverageTier": 20,
                "AccountNumber": 0,
                "BenefitAmount": 11,
                "Address": 2,
                "AddressLine1": 3,
                "BinaryResponse": 15,
                "Carrier": 16,
                "Action": 1,
                "ProductPlanNameOrCode": 56,
                "EmploymentStatus": 28,
                "GenericDate": 34,
                "Zip": 76,
                "AddressLine2": 4,
                "Country": 18,
                "AddressLine3": 5,
                "AdjustDeductAmt": 6,
                "BenefitClass": 12,
                "CoverageAmount": 19,
                "CurrencySalary": 21,
                "FullName": 32,
                "BillingDivision": 14,
                "PhoneNumber": 52,
                "OrgName": 51,
                "BenefitPercentage": 13,
                "SSN": 60,
                "MiddleInitial": 45,
                "Relationship": 59,
                "BeneficiaryType": 10,
                "Product": 55,
                "TimeFreq": 68,
                "EligGroup": 26,
                "Occupation": 50,
                "NumDependents": 48,
                "AgeDays": 7,
                "InforceInd": 42,
                "GuaranteeIssueInd": 37,
                "PremiumFreq": 54,
                "Provider": 57,
                "WaiveReason": 74,
                "USCity": 70,
                "TerminationReasonCode": 67,
                "CompanyCode": 17,
                "GroupNumber": 36,
                "IDNumber": 39,
                "NotesOrDesc": 47,
                "USCounty": 71,
                "HireDate": 38,
                "DriversLicense": 24,
                "MiddleName": 46,
                "USStateCode": 72,
                "DisabilityInd": 23,
                "TobaccoUserOrSmokerInd": 69,
                "IDType": 40,
                "SeqNumber": 64,
                "emailAddress": 77,
                "SalaryFreq": 63,
                "TaxStatus": 65,
                "Reason": 58,
                "MaritalStatus": 44,
                "SalaryEffectiveDate": 62,
                "Units": 73,
                "WorkLocation": 75,
                "AgeGroup": 8,
                "FLSAStatus": 29,
                "FSAamount": 30,
                "IdType": 41,
            },
        },
    }

    max_seq_len = model_config.get("MAX_SEQ_LEN")
    pad = model_config.get("PADDING")

    # load wordPiece tokenizer
    tokenizer = pkg_resources.resource_stream("model_ffm", "data/bert_wp_tok_updated.joblib")
    bert_wp_loaded = joblib.load(tokenizer)

    # load trained model
    fp = pkg_resources.resource_filename("model_ffm", "data/lstm_tuned_nov27.h5")
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

    res = []
    for x, y in zip(pred_labels, confidences):
        res.append([(a, b) for a, b in zip(x, y)])

    return [{"entityId": entity, "predictedResult": prediction}for entity, prediction in zip(all_columns, res)]


# to be deleted..eventually

# payload = {"batch_data": ["Dependent CHILD #3 SSN",
#     "Child#1 DOB",
#     "Child 2 DOB",
#     "Ch1.LastName",
#     "ACC Effective Date"]}
# print(predict(model_name="model_ffm",artifacts=["data/bert_wp_tok_updated.joblib"],model_path="data/lstm_tuned_nov27.h5",inputs=payload))
