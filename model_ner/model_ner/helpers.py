import pandas as pd
import numpy as np
import boto3
import joblib
import tempfile
import spacy

def download_folder_from_s3(bucket_name, s3_folder):
    bucket = boto3.resource("s3").Bucket(bucket_name)
    with tempfile.TemporaryDirectory() as dirpath:
        for obj in bucket.objects.filter(Prefix=s3_folder):
            print(obj.key)
            if obj.key[-1] == '/':
                continue
            with open(f"{dirpath}/{obj.key.split('/')[-1]}", "wb") as file_data:
                bucket.download_fileobj(obj.key, file_data)


def extract_entities_into_df(desc_note) -> pd.DataFrame:
    ner_model = spacy.load("data")
    docs = ner_model(desc_note)
    # print(docs.ents)
    pred_dict = {"system_organ_site": [],
                     "diagnosis_name": [],
                     "direction": [],
                     "acuity": [],
                     "procedure": [],
                     "test": [],
                     "generic_name": [],
                     "name": []}
    for ent in docs.ents:
        if ent.label_ == "SYSTEM_ORGAN_SITE":
            pred_dict["system_organ_site"].append(ent.text)
        elif ent.label_ == "DIRECTION":
            pred_dict["direction"].append(ent.text)
        elif ent.label_ == "DX_NAME":
            pred_dict["diagnosis_name"].append(ent.text)
        elif ent.label_ == "ACUITY":
            pred_dict["acuity"].append(ent.text)
        elif ent.label_ == "PROCEDURE_NAME":
            pred_dict["procedure"].append(ent.text)
        elif ent.label_ == "TEST_NAME":
            pred_dict["test"].append(ent.text)
        elif ent.label_ == "GENERIC_NAME":
            pred_dict["generic_name"].append(ent.text)
        elif ent.label_ == "NAME":
            pred_dict["name"].append(ent.text)
    print(pred_dict)
    return pred_dict