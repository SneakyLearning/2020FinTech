import pandas as pd
import numpy as np
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn import svm
from sklearn.metrics import accuracy_score,roc_auc_score
import lightgbm as lgb

params = {'boosting_type':'gbdt','objective': 'binary','metric': {'binary_logloss', 'auc'},'num_leaves': 8,'max_depth': 8,'min_data_in_leaf': 450,'learning_rate': 0.1,'feature_fraction': 0.9,'bagging_fraction': 0.95,'bagging_freq': 5,'lambda_l1': 1, 'lambda_l2': 0.001,'min_gain_to_split': 0.2,'verbose': 5,'is_unbalance': True}

tag = pd.read_csv(r'D:\MOT\工行数据赛道\new_data\训练数据集\训练数据集\训练数据集_tag.csv')
beh = pd.read_csv(r'D:\MOT\工行数据赛道\new_data\训练数据集\训练数据集\训练数据集_beh.csv')
trd = pd.read_csv(r'D:\MOT\工行数据赛道\new_data\训练数据集\训练数据集\训练数据集_trd.csv')
#sums = pd.read_csv(r'D:\MOT\工行数据赛道\new_data\训练数据集\训练数据集\sum.csv',index_col=0)

tag_results = pd.read_csv(r'D:\MOT\工行数据赛道\new_data\评分数据集b\b\评分数据集_tag_b.csv')
beh_results = pd.read_csv(r'D:\MOT\工行数据赛道\new_data\评分数据集b\b\评分数据集_beh_b.csv')
trd_results = pd.read_csv(r'D:\MOT\工行数据赛道\new_data\评分数据集b\b\评分数据集_trd_b.csv')

def process_trd(trd):
    trd_id = trd['id'].drop_duplicates()
    trd_id = pd.DataFrame(trd_id)
    trd_id['sum'] = np.nan
    for i in trd_id['id']:
        trd_id.loc[trd_id['id'] == i, 'sum'] = trd[trd['id'] == i][
            'cny_trx_amt'].sum()
    return trd_id

def data_wash(tag,sums):
    tag=tag.drop(['edu_deg_cd','deg_cd','atdd_type'],axis=1)#去掉高度缺失的三列

    tag.replace(r'~', np.nan, inplace=True)#替换其他形式的空值
    tag.replace(r'\N', np.nan, inplace=True)
    tag.replace(r'\\N', np.nan, inplace=True)

    df_numeric = tag.select_dtypes(include=[np.number])#填充数值型缺失值
    numeric_cols = df_numeric.columns.values
    for col in numeric_cols:
        missing = tag[col].isnull()
        num_missing = np.sum(missing)
        if num_missing > 0:  # only do the imputation for the columns that have missing values.
            med = tag[col].median()
            tag[col] = tag[col].fillna(med)

    df_non_numeric = tag.select_dtypes(exclude=[np.number])#填充非数值型缺失值
    non_numeric_cols = df_non_numeric.columns.values
    for col in non_numeric_cols:
        missing = tag[col].isnull()
        num_missing = np.sum(missing)
        if num_missing > 0:  # only do the imputation for the columns that have missing values.
            top = tag_results[col].describe()[
                'top']  # impute with the most frequent value.
            tag[col] = tag[col].fillna(top)

    for i in ['gdr_cd', 'mrg_situ_cd', 'acdm_deg_cd']:#特征编码
        mapping = {label: index for index, label in
                   enumerate(set(tag[i]))}
        tag[i] = tag[i].map(mapping)

    for i in tag.columns[1:]:#数值类型转换
        tag[i] = tag[i].astype('int64')

    tag = pd.merge(tag,sums,how='left',on='id')#添加trd中的特征

    x_train = tag.drop('id',axis=1)
    #smo = SMOTE(random_state=16)
    #x_train,y_train = smo.fit_sample(x_train,y_train)#smote处理不平衡
    return x_train

def data_process(x):
    transfer = StandardScaler()
    #pca = PCA(n_components=0.95)
    x = transfer.fit_transform(x)
    #x = pca.fit_transform(x)
    return x,transfer

def train_mode1(x,y):
    lgb_train = lgb.Dataset(x,y)
    gbm = lgb.train(params,lgb_train)

    return gbm

if __name__ == '__main__':
    sums = process_trd(trd)
    y = tag['flag']
    x = data_wash(tag.drop('flag',axis=1),sums)
    x = x.fillna(0)#trd缺失值填0
    x,transfer = data_process(x)
    #训练集已生成

    sums_predict = process_trd(trd_results)
    id_predict = tag_results['id']
    x_predict = data_wash(tag_results,sums_predict)
    print(x_predict)
    x_predict = x_predict.fillna(0)
    x_predict = transfer.transform(x_predict)

    #x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=66)
    #x_train,transfer = data_process(x_train)
    #x_test = transfer.transform(x_test)

    #x_predict = pca.transform(x_predict)

    estimator = train_mode1(x,y)
    y_predict = estimator.predict(x_predict)
    #print(roc_auc_score(y_test,y_predict))
    y_predict = pd.DataFrame(y_predict)

    result = pd.concat([id_predict, y_predict], axis=1)
    result.to_csv("result_lgb_with_trd.txt", sep='\t', index=False, header=None,encoding='utf-8')