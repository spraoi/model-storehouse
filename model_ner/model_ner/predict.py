def predict(**kwargs):
    import spacy
    import pkg_resources


    claim_id = kwargs["inputs"]["Claim Identifier"]
    notes_desc = kwargs["inputs"]["Primary Diagnosis Desc"]

    # spacy needs the model files in a particular folder structure. simpler to have it inside /data than download from S3 into multiple temp directories

    model = pkg_resources.resource_filename(pkg_resources.Requirement.parse("model_ner"), "model_ner/data")

    ner_model = spacy.load(model)
    docs = ner_model(notes_desc)
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

    return [
        {"inputDataSource": f"{claim_id}:0", "entityId": claim_id,
         "predictedResult": pred_dict}]


# print(
#     predict(
#         model_name="NER_Model",
#         artifact=[
#         {
#             "dataId": "5ad3e9c0-b248-4397-89a9-f44f3d3b7454",
#             "dataName": "model_folder",
#             "dataType": "artifact",
#             "dataValue": "s3://demo-temp-ner/test_spacy",
#             "dataValueType": "str"
#         }
#     ],
#         inputs={
#             "Primary Diagnosis Desc": "Primary Diagnosis Desc': 'Unilateral primary osteoarthritis, right knee",
#             "Claim Identifier": "GDC-46016"
#         },
#     )
# )
