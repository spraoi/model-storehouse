def predict(**kwargs):
    import logging
    import random
    import pkg_resources
    import joblib
    import tensorflow as tf

    # Example of loading a local joblib file
    f = pkg_resources.resource_stream("model_abc", "data/bert_wp_tok_updated.joblib")
    j = joblib.load(f)

    # Example of loading a local h5 file
    fp = pkg_resources.resource_filename("model_abc", "data/lstm_tuned_nov27.h5")
    m = tf.keras.models.load_model(fp)

    logging.info(f"{kwargs=}")

    random.seed()
    predictions = [
        {"entityId": str(x), "predictedResult": [(random.random(), r) for r in range(10)]} for x in range(10)
    ]

    return predictions
