
def predict(payload: dict, model_path: str, artifacts: list, parameters: dict):

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from inflection import humanize, underscore
    import joblib
    import logging


    MAX_LEN = parameters.get("model_config").get("MAX_SEQ_LEN")
    pad = parameters.get("model_config").get("PADDING")

    # load wordPiece tokenizer
    tokenizer_path = artifacts[0]
    bert_wp_loaded = joblib.load(tokenizer_path)
    # load trained model
    model_path = model_path
    loaded_model = tf.keras.models.load_model(model_path)

    # load categories:index mappings
    inv_prod_dict = parameters.get("label_mapping").get("product_map")
    prod_dict = {v: k for k, v in inv_prod_dict.items()}

    inv_head_dict = parameters.get("label_mapping").get("header_map")
    head_dict = {v: k for k, v in inv_head_dict.items()}

    inv_party_dict = parameters.get("label_mapping").get("entity_map")
    party_dict = {v: k for k, v in inv_party_dict.items()}

    all_columns = payload.get("batch_data")
    if not all_columns:
        print("no input data !")
    token_ids_test = []
    processed_cols = []
    for column in all_columns:
        column = humanize(underscore(column))
        processed_cols.append(column)
        token_ids_test.append(bert_wp_loaded.encode(column).ids)

    test_tokens = pad_sequences(token_ids_test,
                                padding=pad,
                                value=bert_wp_loaded.get_vocab()['[PAD]'],
                                maxlen=MAX_LEN)

    preds_test = loaded_model.predict(test_tokens)

    p1_test = np.argmax(preds_test[0], axis=1)
    p2_test = np.argmax(preds_test[1], axis=1)
    p3_test = np.argmax(preds_test[2], axis=1)

    prod_label = np.array([prod_dict[x] for x in p1_test]).reshape((-1, 1))
    header_label = np.array([head_dict[x] for x in p2_test]).reshape((-1, 1))
    entity_label = np.array([party_dict[x] for x in p3_test]).reshape((-1, 1))

    prob_1 = tf.nn.softmax(preds_test[0], axis=1)
    prob_2 = tf.nn.softmax(preds_test[1], axis=1)
    prob_3 = tf.nn.softmax(preds_test[2], axis=1)

    prod_confidence = np.array([round(float(x), 4) for x in list(np.max(prob_1, axis=1))]).reshape((-1, 1))
    header_confidence = np.array([round(float(x), 4) for x in list(np.max(prob_2, axis=1))]).reshape((-1, 1))
    entity_confidence = np.array([round(float(x), 4) for x in list(np.max(prob_3, axis=1))]).reshape((-1, 1))

    pred_labels = np.hstack((prod_label,
                             header_label, entity_label)).tolist()
    confidences = np.hstack((prod_confidence, header_confidence,
                             entity_confidence)).tolist()

    res = []
    for x, y in zip(pred_labels, confidences):
        res.append([(a, b) for a, b in zip(x, y)])

    predictions = []
    for entity, prediction in zip(all_columns, res):
        predictions.append({"entityId": entity, "predictedResult": prediction})


    return predictions


#to be deleted..eventually

# payload = {"batch_data": ["Dependent CHILD #3 SSN",
#     "Child#1 DOB",
#     "Child 2 DOB",
#     "Ch1.LastName",
#     "ACC Effective Date"]}
#
# with open("data/MLM_config.json") as json_file:
#     import json
#     config = json.load(json_file)
#     print(predict(payload, "data/lstm_tuned_nov27.h5",["data/bert_wp_tok_updated.joblib"],config))