from celldb import client
from celldb import pd as cpd

from sklearn.decomposition import PCA

connection = client.connect('10.50.101.122')
cohorts = list(client.list_cohorts(connection))



def get_samples(cohort_id):
    return list(client.get_cohort(connection, cohort_id))

def get_features(feature_set_id):
    return list(client.get_feature_set(connection, feature_set_id))

def project_cohort(cohort_id, projection_name, dimensions, function):
    sample_list = get_samples(cohort_id)
    feature_set_id = "{}:{}".format(cohort_id, projection_name)
    feature_names = ["{}_{}".format(projection_name, i) for i in range(dimensions)]
    client.upsert_feature_set(connection, feature_set_id, feature_names)
    df = cpd.df(connection, sample_list, feature_list)
    projected = function(df)
    client.upsert_samples(connection, sample_list, feature_names, projected)
    return feature_set_id

def pca_project_df(n_components):
    return lambda x: PCA(n_components=n_components).fit_transform(x.fillna(0))

import sys

feature_sets = list(client.list_feature_sets(connection))
feature_list = get_features('716a1c2ecc6c546f964f3177af6fb6f1')

feature_set_id = project_cohort(sys.argv[1], 'pca_2', 2, pca_project_df(2))

print(feature_set_id)
