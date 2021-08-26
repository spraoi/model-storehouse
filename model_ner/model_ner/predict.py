def predict(**kwargs):
    import joblib
    import json
    import warnings
    import pandas as pd
    from typing import List
    import boto3
    import os
    import re
    import string
    import random
    import string
    import spacy
    from model_ner.helpers import (download_folder_from_s3,
                                   extract_entities_into_df)


    # download_folder_from_s3("demo-temp-ner","test_spacy")



    # df = pd.read_csv("s3://spr-barrel-qa-datasets/ner_test/synthetic_diagnosis_desc.csv")
    claim = {'Claim Identifier': 1, 'Primary Diagnosis Desc': 'The patient was diagnosed with strain of flexor muscle, fascia and tendon of unspecified finger at wrist and hand level and showed symptoms of post-mixed pain, difficulty in falling from sitting or falling into a supine position, and low body temperature. The patient was discharged without a pulse or any treatment. A few hours after the initial exam the patient reported feeling strong pain in his right wrist and ankle, numbness in his left ankle, headache, and numbness in his right hand.'}
    extract_entities_into_df(claim["Primary Diagnosis Desc"])
    # row = df.to_dict(orient="records")[0]
    # print(row)
    # df = pd.DataFrame


predict()
