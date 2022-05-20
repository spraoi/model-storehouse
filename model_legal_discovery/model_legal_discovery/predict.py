def predict(**kwargs):
    import os
    import functools
    import json
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
    from nltk import tokenize
    import pandas as pd
    import profanity_check as pfc
    import PyPDF2
    import torch

    # from airflow.configuration import conf
    # from airflow.hooks.spraoi_hooks import ElasticSearchHook
    from numpy import sign
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    inputs = kwargs.get("inputs")
    logging.info(f"{inputs=}")

    # todo: Move inside Airflow operator?
    # index = conf.get("dagger", "case_model_es_index")
    # es_hook = ElasticSearchHook(elasticsearch_conn_id="elasticsearch")

    def download_s3_folder(bucket_name, s3_folder, local_dir=None):
        bucket = boto3.resource("s3").Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=s3_folder):
            target = (
                obj.key
                if local_dir is None
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            )
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == "/":
                continue
            bucket.download_file(obj.key, target)

    def rec_to_actions(df):

        for record in df.to_dict(orient="records"):
            try:
                res = es_hook.create_or_update(
                    index=index,
                    body=json.dumps(record),
                    doc_id=record["id"],
                )
                logging.info(f"{res=}")
            except BaseException as e:
                logging.error("Failed to push data to ElasticSearch [{}]".format(e))

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

    # setup location for NLTK data

    transformers_data = pkg_resources.resource_filename(
        pkg_resources.Requirement.parse("model_legal_discovery"),
        "model_legal_discovery/transformers",
    )
    ntlk_data_loation = pkg_resources.resource_filename(
        pkg_resources.Requirement.parse("model_legal_discovery"),
        "model_legal_discovery/nltk_data",
    )
    nltk.data.path.append(ntlk_data_loation)

    # populate folders using data from s3
    download_s3_folder("legal-disc", "nltk_data/", ntlk_data_loation)
    download_s3_folder("legal-disc", "transformers/", transformers_data)

    os.putenv("NLTK_DATA", ntlk_data_loation)

    tokenizer_load = AutoTokenizer.from_pretrained(transformers_data)
    model_load = AutoModelForSequenceClassification.from_pretrained(transformers_data)

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
        #     logging.info("Initial Document ES Load")
        #     rec_to_actions(pd.DataFrame([doc_dict]))  # todo:
        #
        logging.info(f"download document{document.get('keyPath')}")
        with tempfile.NamedTemporaryFile() as fp:
            bucket = boto3.resource("s3").Bucket(document.get("bucket"))
            bucket.download_fileobj(document.get("keyPath"), fp)
            file_reader = PyPDF2.PdfFileReader(fp)
            all_text = "".join(
                [
                    file_reader.getPage(page_num).extractText()
                    for page_num in range(0, file_reader.numPages)
                ]
            )
            total_pages = file_reader.numPages
        logging.info(f"{total_pages=}")
        doc_dict["text"] = all_text
        doc_dict["totalPages"] = total_pages
        logging.info("Get Tokens")
        tokens = tokenize.sent_tokenize(all_text)  # nltk
        #
        logging.info("Get sentence dict and row list")
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
        #
        logging.info("Get status")
        total_words = all_text.split()
        no_words = len(total_words)
        total_sentences = tokens
        no_sentences = len(total_sentences)
        #
        logging.info("Update doc dict")
        doc_dict["totalWords"] = no_words
        doc_dict["totalSentences"] = no_sentences
        doc_dict["sentences"] = sentence_row_list
        if sentence_row_list:
            doc_dict["profanityCheck"] = int(
                round(max((x["profanityCheck"]) for x in sentence_row_list), 0)  # noqa
            )
        else:
            doc_dict["profanityCheck"] = 0

        logging.info("Entity Extract")
        doc_dict["tags"], doc_dict["ctext"] = entity_extract(all_text)
        if tokens:
            logging.info("tokens found")
            overall_score = round(overall_score / len(tokens), 3)
            doc_dict["sentimentFeedback"] = sign(overall_score) * ceil(
                abs(overall_score)
            )
            doc_dict["group"] = None
            logging.info("build word row list")
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
            logging.info("Create doc dataframe")
            doc_df = pd.DataFrame(
                [
                    doc_dict,
                ]
            )
            if doc_df.shape[0]:
                logging.info("Append doc dataframe to case dataframe")
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
    #     logging.info("Document ES Update")
    #     rec_to_actions(pd.DataFrame([doc_dict]))  # todo:
    if len(documents) > 3:
        logging.info("Get Cluster")
        case_df = get_cluster(case_df)
    case_df.drop(columns=["text", "ctext"], inplace=True)
    # logging.info("Dataframe document ES Load")
    # rec_to_actions(case_df)  # todo:
    #
    return results


# print(
#     predict(
#         inputs= {
#         "caseId": 'spr:bz:case::0f918737-8332-4033-9478-3bc774e703e5',
#         "documents": [
#           {
#             "documentId": 'spr:bz:document::20007992-6cde-4cbe-b6cd-8e77d684819a',
#             "documentName": 'abc-1',
#             "status": 'pending',
#             "bucket": 'spr-barrel-dev-nextra-documents',
#             "keyPath": 'upload/spr:bz:case::0ea2ab20-f6d8-4326-86a5-7cba526a0fcc/1629126547439/doc3_1.pdf',
#             "createdAt": 1626960092552,
#             "updatedAt": 1626960092552,
#           },
#           {
#             "documentId": 'spr:bz:document::20007992-6cde-4cbe-b6cd-8e77d684819a',
#             "documentName": 'def-1',
#             "status": 'pending',
#             "bucket": 'spr-barrel-dev-nextra-documents',
#             "keyPath": 'jd-test/SentimentDocument1_Redacted.pdf',
#             "createdAt": 1626960092552,
#             "updatedAt": 1626960092552,
#           },
#         ],
#     }
#     )
# )

# """
#     Prediction function for the Legal Discovery model
#     Example input:
#         {
#         caseId: 'spr:bz:case::0f918737-8332-4033-9478-3bc774e703e5',
#         documents: [
#           {
#             documentId: 'spr:bz:document::20007992-6cde-4cbe-b6cd-8e77d684819a',
#             documentName: 'abc-1',
#             status: 'pending',
#             bucket: 'spr-barrel-dev-nextra-documents',
#             keyPath: 'spr:bz:case::0f918737-8332-4033-9478-3bc774e703e5/abc-1.pdf',
#             createdAt: 1626960092552,
#             updatedAt: 1626960092552,
#           },
#           {
#             documentId: 'spr:bz:document::20007992-6cde-4cbe-b6cd-8e77d684819a',
#             documentName: 'def-1',
#             status: 'pending',
#             bucket: 'spr-barrel-dev-nextra-documents',
#             keyPath: 'spr:bz:case::0f918737-8332-4033-9478-3bc774e703e5/def-1.pdf',
#             createdAt: 1626960092552,
#             updatedAt: 1626960092552,
#           },
#         ],
#     }
#
# """
#
