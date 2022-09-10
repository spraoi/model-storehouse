"""
 Predict function for legal discovery
"""


def predict(**kwargs):
    import os
    import functools
    import logging
    import re
    import tempfile
    import time
    from collections import Counter
    from math import ceil
    from string import punctuation
    import pkg_resources
    import boto3
    import en_core_web_sm
    import nltk
    from nltk import tokenize
    import pandas as pd
    import profanity_check as pfc
    import PyPDF2
    import torch
    from numpy import sign
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    inputs = kwargs.get("inputs")

    def download_s3_res(
        bucket_name: str,
        key_name: str = None,
        s3_folder: str = None,
        local_dir: str = None,
        temp_file: str = None,
    ):
        bucket = boto3.resource("s3").Bucket(bucket_name)
        if s3_folder:
            for obj in bucket.objects.filter(Prefix=s3_folder):
                target = (
                    obj.key
                    if local_dir is None
                    else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
                )
                logging.info(f"{target=}")
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                if obj.key[-1] == "/":
                    continue
                bucket.download_file(obj.key, target)
        else:
            bucket.download_fileobj(key_name, temp_file)

    def get_cluster(case_dataframe):
        all_docs = case_dataframe["text"].to_list()

        vectorizer = TfidfVectorizer(stop_words={"english"})
        all_docs_transformed = vectorizer.fit_transform(all_docs)
        model = KMeans(n_clusters=3, init="k-means++", max_iter=200, n_init=10)
        model.fit(all_docs_transformed)
        labels = model.labels_
        case_dataframe["group"] = labels
        return case_dataframe

    def entity_extract(text_given):

        nlp = en_core_web_sm.load()
        result = []
        pos_tag = ["PROPN", "ADJ", "NOUN"]
        doc = nlp(str(text_given).lower())
        for doc_token in doc:
            if (
                doc_token.text in nlp.Defaults.stop_words
                or doc_token.text in punctuation
            ):
                continue
            if doc_token.pos_ in pos_tag:
                result.append(doc_token.text)
        hashtags = [
            ("#" + x[0])
            for x in Counter(result).most_common(5)
            if all(c.isalnum() for c in x[0])
        ]
        return hashtags, [str(text_given).lower()]

    def sentiment_score(tokenizer, model, review):
        token_list = tokenizer.encode(
            review, return_tensors="pt", max_length=512, truncation=True
        )
        result = model(token_list)
        return int(torch.argmax(result.logits)) - 2

    def get_bucket_and_key_from_s3_uri(uri):
        bucket, key = uri.split("/", 2)[-1].split("/", 1)
        return bucket, key

    ntlk_data_loation = pkg_resources.resource_filename(
        pkg_resources.Requirement.parse("model_legal_discovery"),
        "model_legal_discovery/nltk_data",
    )

    for artifact in kwargs.get("artifact"):
        if artifact.get("dataName") == "model_dir":
            model_bucket, model_folder = get_bucket_and_key_from_s3_uri(
                artifact.get("dataValue")
            )
            break
    else:
        # model dir not found
        raise ValueError("Missing artifact called model_dir")

    tmp_dir = tempfile.gettempdir()
    # populate folders using data from s3
    download_s3_res(bucket_name=model_bucket, s3_folder=model_folder, local_dir=tmp_dir)
    # add search path to nltk
    nltk.data.path.append(ntlk_data_loation)

    tokenizer_load = AutoTokenizer.from_pretrained(tmp_dir)
    model_load = AutoModelForSequenceClassification.from_pretrained(tmp_dir)
    #
    case_df = pd.DataFrame()
    case_id = inputs.get("caseId")

    documents = inputs.get("documents")

    loaded_sentiment_score = functools.partial(
        sentiment_score, tokenizer_load, model_load
    )

    results = []
    for document in documents:
        doc_dict = dict(
            id=document.get("documentId"),
            caseId=case_id,
            documentName=document.get("documentName"),
            documentId=document.get("documentId"),
            bucket=document.get("bucket"),
            keyPath=document.get("keyPath"),
            documentStatus="In-Progress",
            createdAt=document.get("createdAt"),
        )

        with tempfile.NamedTemporaryFile() as fp:
            download_s3_res(
                bucket_name=document.get("bucket"),
                key_name=document.get("keyPath"),
                temp_file=fp,
            )
            file_reader = PyPDF2.PdfFileReader(fp)
            all_text = "".join(
                [
                    file_reader.getPage(page_num).extractText()
                    for page_num in range(0, file_reader.numPages)
                ]
            )
            total_pages = file_reader.numPages
        doc_dict["text"] = all_text
        doc_dict["totalPages"] = total_pages
        tokens = tokenize.sent_tokenize(all_text)  # nltk
        overall_score = 0
        sentence_row_list = []
        for token in tokens:
            score = loaded_sentiment_score(token)
            sentence_dict = dict(
                sentence=token,
                sentimentFeedback=score,
                profanityCheck=pfc.predict_prob([token])[0],
            )
            overall_score += score
            sentence_row_list.append(sentence_dict)

        total_words = all_text.split()
        no_words = len(total_words)
        total_sentences = tokens
        no_sentences = len(total_sentences)

        doc_dict["totalWords"] = no_words
        doc_dict["totalSentences"] = no_sentences
        doc_dict["sentences"] = sentence_row_list
        if sentence_row_list:
            doc_dict["profanityCheck"] = int(
                round(max((x["profanityCheck"]) for x in sentence_row_list), 0)  # noqa
            )
        else:
            doc_dict["profanityCheck"] = 0

        doc_dict["tags"], doc_dict["ctext"] = entity_extract(all_text)
        if tokens:
            overall_score = round(overall_score / len(tokens), 3)
            doc_dict["sentimentFeedback"] = sign(overall_score) * ceil(
                abs(overall_score)
            )
            doc_dict["group"] = None
            word_row_list = []
            for tags in doc_dict.get("tags"):
                tag = tags.replace("#", "")
                score = loaded_sentiment_score(tag)
                if score:
                    word_dict = dict(
                        word=tags,
                        sentimentFeedback=score,
                        count=sum(
                            1
                            for _ in re.finditer(
                                r"\b%s\b" % re.escape(tag), doc_dict["ctext"][0]
                            )
                        ),
                    )
                    word_row_list.append(word_dict)
            doc_dict["words"] = word_row_list
            doc_dict["documentStatus"] = "completed"
            doc_dict["updatedAt"] = int(time.time() * 1000.0)
            results.append(
                dict(
                    inputDataSource=case_id,
                    entityId=document.get("documentId"),
                    predictedResult=dict(
                        sentimentFeedback=doc_dict["sentimentFeedback"],
                        profanityCheck=doc_dict["profanityCheck"],
                    ),
                )
            )
            doc_df = pd.DataFrame(
                [
                    doc_dict,
                ]
            )
            if doc_df.shape[0]:
                case_df = case_df.append(doc_df, ignore_index=True)
        else:
            doc_dict["documentStatus"] = "unprocessed"
            doc_dict["updatedAt"] = int(time.time() * 1000.0)
            results.append(
                dict(
                    inputDataSource=case_id,
                    entityId=document.get("documentId"),
                    predictedResult=dict(
                        sentimentFeedback=0,
                        profanityCheck=0,
                    ),
                )
            )

    if len(documents) > 3:
        case_df = get_cluster(case_df)
    case_df.drop(columns=["text", "ctext"], inplace=True)

    return results


# print(
#     predict(
#         artifact=[
#             {
#                 "dataName": "model_dir",
#                 "dataType": "artifact",
#                 "dataValue": "s3://legal-disc/transformers/",
#                 "dataValueType": "str"
#             }],
#         inputs={
#             "caseId": 'spr:bz:case::0f918737-8332-4033-9478-3bc774e703e5',
#             "documents": [
#                 {
#                     "documentId": 'spr:bz:document::20007992-6cde-4cbe-b6cd-8e77d684819a',
#                     "documentName": 'abc-1',
#                     "status": 'pending',
#                     "bucket": 'spr-barrel-dev-nextra-documents',
#                     "keyPath": 'upload/spr:bz:case::0ea2ab20-f6d8-4326-86a5-7cba526a0fcc/1629126547439/doc3_1.pdf',
#                     "createdAt": 1626960092552,
#                     "updatedAt": 1626960092552
#                 },
#                 {
#                     "documentId": 'spr:bz:document::20007992-6cde-4cbe-b6cd-8e77d684819a',
#                     "documentName": 'def-1',
#                     "status": 'pending',
#                     "bucket": 'spr-barrel-dev-nextra-documents',
#                     "keyPath": 'jd-test/SentimentDocument1_Redacted.pdf',
#                     "createdAt": 1626960092552,
#                     "updatedAt": 1626960092552
#                 }
#             ]
#         }
#     )
# )
