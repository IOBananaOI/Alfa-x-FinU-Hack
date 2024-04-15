import pandas as pd
import numpy as np

import joblib

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


def weighted_roc_auc(y_true, y_pred, labels):
    cluster_weights = pd.read_excel("../data/cluster_weights.xlsx").set_index("cluster")
    weights_dict = cluster_weights["unnorm_weight"].to_dict()

    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)


def fill_na_by_start_cluster(df, for_num='mean', for_object='mode'):
    """
    Заполняет пропуски в численных и категориальных признаках,
    согласно переданным стратегиям заполнения, используя start_cluster.
    """

    df['start_cluster'] = df['start_cluster'].fillna('None')

    object_df = df.select_dtypes(include=['object']).copy()
    no_object_df = df.drop(columns=object_df.drop(columns=['start_cluster']).columns)

    if for_num == 'mean':
        no_object_df = no_object_df.groupby('start_cluster').apply(lambda group: group.fillna(group.mean()))

    if for_num == 'mode':
        no_object_df = no_object_df.groupby('start_cluster').apply(lambda group: group.fillna(group.mode().iloc[0]))

    if for_num == 'backfill':
        no_object_df = no_object_df.groupby('start_cluster').apply(lambda group: group.fillna(method='backfill'))

    if for_num == 'bfill':
        no_object_df = no_object_df.groupby('start_cluster').apply(lambda group: group.fillna(method='bfill'))

    if for_num == 'ffill':
        no_object_df = no_object_df.groupby('start_cluster').apply(lambda group: group.fillna(method='ffill'))

    if for_num == 'None':
        no_object_df = no_object_df.fillna('None')

    if for_object == 'mode':
        object_df = object_df.groupby('start_cluster').apply(lambda group: group.fillna(group.mode().iloc[0]))

    if for_object == 'backfill':
        object_df = object_df.groupby('start_cluster').apply(lambda group: group.fillna(method='backfill'))

    if for_object == 'bfill':
        object_df = object_df.groupby('start_cluster').apply(lambda group: group.fillna(method='bfill'))

    if for_object == 'ffill':
        object_df = object_df.groupby('start_cluster').apply(lambda group: group.fillna(method='ffill'))

    if for_object == 'None':
        object_df = object_df.fillna('None')

    result_df = pd.concat([no_object_df, object_df], axis=1)
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def fill_na(df, for_num='mean', for_object='mode'):
    """
    Заполняет пропуски в численных и категориальных признаках,
    согласно переданным стратегиям заполнения.
    """

    no_object_df = df.iloc[:, (df.dtypes != 'object').values]
    object_df = df.iloc[:, (df.dtypes == 'object').values]

    if for_num == 'mean':
        values = no_object_df.mean()
        no_object_df = no_object_df.fillna(values)

    if for_num == 'mode':
        values = no_object_df.mode().iloc[0]
        no_object_df = no_object_df.fillna(values)

    if for_num == 'backfill':
        no_object_df = no_object_df.fillna(method='backfill')

    if for_num == 'bfill':
        no_object_df = no_object_df.fillna(method='bfill')

    if for_num == 'ffill':
        no_object_df = no_object_df.fillna(method='ffill')

    if for_object == 'mode':
        values = object_df.mode().iloc[0]
        object_df = object_df.fillna(values)

    if for_object == 'backfill':
        object_df = object_df.fillna(method='backfill')

    if for_object == 'bfill':
        object_df = object_df.fillna(method='bfill')

    if for_object == 'ffill':
        object_df = object_df.fillna(method='ffill')

    if for_num == 'None':
        no_object_df = no_object_df.fillna('None')

    if for_object == 'None':
        object_df = object_df.fillna('None')

    return no_object_df.reset_index().merge(object_df.reset_index(), left_on='index', right_on='index').drop(
        columns=['index'])


def reshape_dataset(df, drop_month_3=True, transform_categories=True):
    """
    Группирует данные по пользователям, и вытягивает по месяцам.
    Т.е. (N, D) --> (N//3, D*3), где N, D - кол-во строк и столбцов соотв.
    """

    # Меняем типы, для более быстрых вычислений
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Группируем строки датафрейма по пользователям
    df_pivot = df.pivot_table(index="id", columns="date", aggfunc='first')

    # Задаём новые имена столбцам, основываясь на месяце
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    df_pivot = df_pivot.reset_index()

    # Преобразовываем категориальные признаки к типу pandas Category
    if transform_categories:
        category_columns = df_pivot.select_dtypes(include=['category', 'O']).columns
        df_pivot[category_columns] = df_pivot[category_columns].astype("category")

    # Избавляемся от start_cluster_month_3
    if drop_month_3:
        df_pivot = df_pivot.drop(columns=['start_cluster_month_3'])

    return df_pivot


def weighted_roc_auc(y_true, y_pred, labels):
    """
    Взвешенный roc_auc
    """

    cluster_weights = pd.read_excel("data/cluster_weights.xlsx").set_index("cluster")
    weights_dict = cluster_weights["unnorm_weight"].to_dict()

    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)


def validate(X,y,model,n_splits=5): 
    """ 
    Валидация модели при помощи различных подходов. 
    """ 
    scores=[] 
    kf = StratifiedKFold(n_splits, shuffle=True, random_state=42) 
    for train_index, test_index in kf.split(X=X, y=y): 
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
        y_train, y_test = y.iloc[train_index], y.iloc[test_index] 
        model.fit(X_train,y_train) 
        y_pred=model.predict_proba(X_test) 
        score=weighted_roc_auc(y_test, y_pred, labels=model.classes_) 
        scores.append(score) 
    return np.array(scores).mean()



def fill_start_cluster(test_df):
    # Get pretrained model
    model = joblib.load('weights/fill_cluster_model_weights.pkl')

    # Select columns from dataframe
    cols = ['start_cluster_month_1', 'start_cluster_month_2']
    prev_clusters = test_df[cols]

    # Transform features to numbers
    le = LabelEncoder()
    le.fit(prev_clusters['start_cluster_month_1'])

    prev_clusters['start_cluster_month_1'] = le.transform(prev_clusters['start_cluster_month_1'])
    prev_clusters['start_cluster_month_2'] = le.transform(prev_clusters['start_cluster_month_2'])

    # Predict clusters for 6th month
    predicted_clusters = model.predict(prev_clusters)

    predicted_clusters = pd.Series(le.inverse_transform(predicted_clusters))

    # predicted_clusters = predicted_clusters.astype('category')

    test_df['start_cluster_month_3'] = predicted_clusters

    return test_df
