# from typing import List, Tuple, Dict
# import json
#
#
# import dask.dataframe as dd
# from dask.distributed import LocalCluster
# from dask.distributed import Client, Security
# # from boogie.models.dataset import *
# # from boogie.workers.bagloader import BagLoader
# # from boogie.models import FileFormat
# # from boogie.models import S3Path
#
#
# def get_model_info(mv_data_url: str, client_id: str, user_id):
#
#     """
#     get info from model-version endpoint
#     needs: complete endpoint url including the host
#     returns: requests.get response
#     """
#     headers = {
#         "Content-Type": "application/json",
#         "x-spr-client-id": client_id,
#         "x-spr-user-id": user_id
#     }
#
#     return requests.get(url=mv_data_url, headers=headers)
#
#
# def fetch_artifact_key(host, MlModelId, version_num,
#                        artifactName) -> Tuple[str, str]:
#     """
#     fetch the s3 key of the artifacts based on the artifact type
#     type can be "model_key" or "template_key"
#     """
#
#     mv_data_end_pt = f"/ml-model-service/api/v1/model/{MlModelId}/version/{version_num}"
#     mv_data_url = f"{host}{mv_data_end_pt}"
#     response = get_model_info(mv_data_url, client_id, user_id)
#
#     res = response.json()
#
#     for artifact in res.get('artifact'):
#         if artifact.get('dataName') == artifactName:
#             return artifact.get('dataValue')
#
#
# class DecimalEncoder(json.JSONEncoder):
#     def default(self, o):
#         if isinstance(o, decimal.Decimal):
#             return float(o)
#         return super(DecimalEncoder, self).default(o)
#
#
# def extract_top_drivers(df_to_explain, shap_values):
#     import numpy as np
#     import pandas as pd
#     """
#     explain predictions on a dataframe using SHAP explainer
#     Needs: the input df, shapley values output
#     from the shap exlainer.
#     Returns: shapley df containing
#     """
#
#     features = df_to_explain.columns
#     top_drivers = []
#     impacts = []
#     for i in range(df_to_explain.shape[0]):
#         id_sorted = np.argsort(shap_values[i].values)
#         top5_positive = features[id_sorted[:-6:-1]]
#         top2_negative = features[id_sorted[:2]]
#         top5_positive_data = shap_values[i].data[id_sorted[:-6:-1]]
#         top2_negative_data = shap_values[i].data[id_sorted[:2]]
#         top5_pos_imp = np.round(shap_values[i].values[id_sorted[:-6:-1]], 4)
#         top2_neg_imp = np.round(shap_values[i].values[id_sorted[:2]], 4)
#         top5_info = [
#             x + " = " + str(y)
#             for x, y in zip(top5_positive, top5_positive_data)
#         ]
#         top2_info = [
#             x + " = " + str(y)
#             for x, y in zip(top2_negative, top2_negative_data)
#         ]
#         top_drivers.append(list(top5_info) + list(top2_info))
#         impacts.append(list(top5_pos_imp) + list(top2_neg_imp))
#
#     drivers_df = pd.DataFrame({'drivers': top_drivers})
#     impacts_df = pd.DataFrame({'impacts': impacts})
#
#     topdrivers_df = pd.DataFrame(
#         drivers_df['drivers'].to_list(),
#         columns=['top_driver_' + str(i) for i in range(1, 8)])
#     top_imp_df = pd.DataFrame(
#         impacts_df['impacts'].to_list(),
#         columns=['lapse_impact' + str(i) for i in range(1, 8)])
#     shapley_df = pd.concat([topdrivers_df, top_imp_df], axis=1)
#
#     return shapley_df
#
#
# def predict_fn(x, model):
#     return model.predict_proba(x).astype(float)
#
# def load_dataframe(dataset_id: str, user_id: str, client_id: str):
#     ds = Dataset.get(dataset_id=dataset_id, user_id=user_id, client_id=client_id)
#     bag = BagLoader().load_bag(dataset=ds)
#     ddf = bag.to_dataframe()
#     return ddf
