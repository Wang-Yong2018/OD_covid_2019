import pandas as pd
from sqlalchemy import create_engine
from sklearn.manifold import TSNE
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
# Import all models
# from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.feature_bagging import FeatureBagging
# from pyod.models.knn import KNN
# from pyod.models.lof import LOF
# from pyod.models.mcd import MCD
# from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
# from pyod.models.lscp import LSCP
FILE_LOCATION = '/home/notebooks/covid_19/DXY-COVID-19-Data/csv'

CSV_FILES = ['DXYArea.csv', 'DXYNews.csv', 'DXYOverall.csv', 'DXYRumors.csv']
url = "postgres://postgres:postgres@192.168.1.200:15432/wh_coronavirus"
engine = create_engine(url)
plt.rcParams['axes.unicode_minus'] = False
FIG_SIZE = (15, 4)


def psql_insert_copy(table, conn, keys, data_iter):
    import csv
    from io import StringIO

    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


def import_dxy(engine, location=FILE_LOCATION, schema='dxy'):
    file_location = location
    csv_files = CSV_FILES
    for file_name in csv_files[:]:
        full_name = "/".join([file_location, file_name])
        print(full_name)
        df = pd.read_csv(full_name)
        table_name = file_name.split(".")[0].lower()
        df.columns = df.columns.str.lower()
        time_cols = ['crawltime', 'updatetime', 'createtime',
                     'modifytime', 'datainfotime', 'pubdate']
        for time_col in time_cols:
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
        df.to_sql(table_name, engine, schema=schema,
                  if_exists='replace', method=psql_insert_copy)


def show_1d(s, log_scale=False):

    fig, ax = plt.subplots(1, 3, figsize=FIG_SIZE)

    tmp_s = s
    data_name = ''
    sns.swarmplot(y=tmp_s, ax=ax[0])
    lag_plot(tmp_s, ax=ax[1], alpha=0.5)
    autocorrelation_plot(tmp_s, ax[2], alpha=0.5)
    if log_scale:

        ax[0].set(yscale='symlog')
        ax[1].set(yscale='symlog', xscale='symlog')

    ax[0].set_title(f"{data_name}总体图")
    ax[1].set_title(f"{data_name}时滞图")
    ax[2].set_title(f"{data_name}自相关图")


def OD_detect(df, id_col=None, contamination=0.05, trans_cols=None):
    """
    use pyod lib to find 5% outlier in dataset
    """
    df = df.copy()
    OD_clfs = {"HBOS": HBOS(contamination=contamination),
               "IForest": IForest(contamination=contamination, max_samples=128),
               "CBLOF": CBLOF(contamination=contamination, n_clusters=5),
               # "OCSVM": OCSVM(contamination=contamination),
               "PCA": PCA(contamination=contamination)}
    results_list = []
    od_cols = ["id", "name", "result", "label"]

    if id_col is None:
        s_id = df.index
        od_cols = df.columns
    else:
        s_id = df[id_col]
        X_cols = df.columns.drop(id_col)

    if trans_cols is not None:
        for col in trans_cols:
            df[col] = PowerTransformer().fit_transform(
                df[col].values.reshape(-1, 1))

    for clf_name, clf in OD_clfs.items():
        od_result = pd.DataFrame(columns=od_cols)  # create an empty  dataframe

        od_result["id"] = s_id

        od_result['name'] = clf_name
        print(f"{clf_name}, {clf}")

        clf.fit(df[X_cols])

        od_result['result'] = clf.decision_scores_
        od_result['label'] = clf.labels_

        results_list.append(od_result)

    od_results_df = pd.concat(results_list, axis=0, ignore_index=True)
    job_name = f'{pd.datetime.now():%H%M}'
    od_results_df['job_name'] = job_name
    od_results_df.to_sql('t_ml', engine, if_exists='append',
                         schema='wh_v1', method=psql_insert_copy)
    print(
        f"OD results {od_results_df.shape}exported to database{engine},job_name={job_name}")
    return od_results_df


def MVOD_detect(df, id_col=None, contamination=0.05, is_transfer=True,is_db=True):
    """
    use pyod lib to find 5% outlier in dataset
    """
    df = df.copy()
    OD_clfs = {"HBOS": HBOS(contamination=contamination),
               "IForest": IForest(contamination=contamination, max_samples=128),
               "CBLOF": CBLOF(contamination=contamination, n_clusters=5),
               # "OCSVM": OCSVM(contamination=contamination),
               "PCA": PCA(contamination=contamination)}
    results_list = []
    od_cols = ["id", "name", "result", "label"]

    if id_col is None:
        s_id = df.index.values
        X_cols = df.columns
    else:
        s_id = df[id_col].values
        X_cols = df.columns.drop(id_col)

    for clf_name, clf in OD_clfs.items():
        od_result = pd.DataFrame(columns=od_cols)  # create an empty  dataframe

        od_result["id"] = s_id

        od_result['name'] = clf_name
        print(f"{clf_name}, {clf}")

        X = df[X_cols].values
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if is_transfer:
            X = PowerTransformer().fit_transform(X)

        clf.fit(X)

        od_result['result'] = clf.decision_scores_
        od_result['label'] = clf.labels_

        results_list.append(od_result)

    od_results_df = pd.concat(results_list, axis=0, ignore_index=True)
    job_name = f'{pd.datetime.now():%H%M}'
    od_results_df['job_name'] = job_name
    if is_db:
        od_results_df.to_sql('t_ml', engine, if_exists='append',
                         schema='wh_v1', method=psql_insert_copy)
        print(
            f"OD results {od_results_df.shape}exported to database{engine},job_name={job_name}")
    return od_results_df


def show_pareto(df, bar_col, val_col, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    s = df.groupby(bar_col)[val_col].sum()
    s = s.sort_values(ascending=False)
    cumpercentage = s.cumsum()/s.sum()*100
    ax1 = s.plot.bar(ax=ax, rot=30)
    # ax1.set_xticklabels(rotation=30)
    ax2 = ax1.twinx()
    cumpercentage.plot(ax=ax2, color="C1", marker="D")
    ax2.yaxis.set_major_formatter(PercentFormatter())

    return


def as_numeric(df, drop_first=False):
    tmp_df = pd.get_dummies(df, drop_first=drop_first)
    tmp_df['publish_date'] = tmp_df['publish_date'].dt.dayofyear
    tmp_df['collection_date'] = tmp_df['collection_date'].dt.dayofyear
    return tmp_df


def show_mvod_result(od_input, od_output, perplexity=3):
    X = od_input
    c_label = od_output['label'].replace({0: "0-正常", 1: "1-异常"})
    clf_list = od_output['name'].value_counts().sort_index().index

    tsne = TSNE(n_components=2, init='random',
                random_state=0, perplexity=perplexity)
    print(tsne)
    tsne_X = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.set_context('notebook')

    ax = od_output.pivot_table(index='name', columns=c_label, values='qty', aggfunc='count').sort_index(
    ).plot.bar(stacked=True, table=True, ax=axes[0])
    ax.get_xaxis().set_visible(False)   # Hide Ticks
    ax.set_title('异常点检出数量比较-按算法')
    ax.legend(loc='lower right')

    ax = sns.swarmplot(data=od_output, y='qty', x='name',
                       hue=c_label, ax=axes[1], order=clf_list)
    ax.set_title('异常检测结果数值比较 - 按算法')
    ax.set_xlabel('疾控数据')
    ax.set_ylabel('异常检测算法')
    ax.legend(loc='upper right')

    ax = sns.swarmplot(data=od_output, y='qty', x='name',
                       hue=c_label, ax=axes[2], order=clf_list)
    ax.set_title('异常检测结果数值(log化)比较 - 按算法')
    ax.set_xlabel('疾控数据')
    ax.set_ylabel('异常检测算法')
    ax.set(yscale='symlog')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots(1, len(clf_list), figsize=(15, 4), sharey=True)

    for i, name in enumerate(clf_list):
        label = od_output.query(f'name=="{name}"')[
            'label'].replace({0: "0-正常", 1: "1-异常"})
        g = sns.scatterplot(
            x=tsne_X[:, 0], y=tsne_X[:, 1], hue=label, ax=ax[i], hue_order=["0-正常", "1-异常"])
        g.set_title(f'{name} 异常检测结果图')

    return


def show_images(loc_list, orient='v', aspect='auto'):
    """
    loc_list is list like following format :
        img_info = [{"name":"异常和噪音的相同之处",
               "loc":'images/anomalies_1.jpg'},
              {"name": "异常和噪音不同之处",
                  "loc":'images/anomalies_2.jpg'}]
    """
    img_info = loc_list
    nimgs = len(img_info)
    if orient == 'v':
        fig, ax = plt.subplots(1, nimgs, figsize=(6*nimgs, 6))
    elif orient == 'h':
        fig, ax = plt.subplots(nimgs, 1, figsize=(6*nimgs, 6))
    for i, img_info in enumerate(img_info):
        img_loc = img_info["loc"]
        img_name = img_info["name"]
        img = imgplt.imread(img_loc)
        if nimgs > 1:
            ax[i].imshow(img, aspect=aspect)
            ax[i].axis('off')
            ax[i].set_title(img_name)
        else:
            ax.imshow(img, aspect=aspect)
            ax.axis('off')
            ax.set_title(img_name)
    return
