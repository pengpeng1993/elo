# https://www.kaggle.com/c/elo-merchant-category-recommendation

import pandas as pd
import logging

# 创建logger对象
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# 创建FileHandler对象
fh = logging.FileHandler('./log/01_date_analysis.log', "w")
fh.setLevel(logging.DEBUG)
# 创建Formatter对象
formatter = logging.Formatter()
fh.setFormatter(formatter)
# 将FileHandler对象添加到Logger对象中
logger.addHandler(fh)

train = pd.read_csv('../data/primeval/train.csv')
test = pd.read_csv('../data/primeval/test.csv')

logger.info(train.shape)  # (201917, 6)
logger.info(test.shape)  # (123623, 5)

pd.set_option('display.max_columns', None)  # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行
logger.info(train.head(5))
logger.info(test.head(5))
print(train.info())
print(test.info())

# logger.info(train['first_active_month'].value_counts())
logger.info(train['feature_1'].value_counts())
logger.info(train['feature_2'].value_counts())
logger.info(train['feature_3'].value_counts())

# 信用卡号 第一次激活月份 特征1,2,3 预测目标值忠诚度评分

# 数据正确性校验
# 数据集id无重复
logger.info(train['card_id'].nunique() == train.shape[0])  # True
logger.info(test['card_id'].nunique() == test.shape[0])  # True
logger.info(test['card_id'].nunique() + train['card_id'].nunique() == len(
    set(test['card_id'].values.tolist() + train['card_id'].values.tolist())))  # True

logger.info(train.isnull().sum())
logger.info(test.isnull().sum())  # first_active_month    1

# 缺失一条记录测试集缺失一条激活月份的记录

# 异常值
logger.info(train['target'].describe())  #

import seaborn as sns
import matplotlib.pyplot as plt

#
sns.set()
sns.histplot(train['target'], kde=True)
plt.show()

logger.info((train['target'] < -30).sum())  # 2207

# 对于连续变量服从正态分布 一般可以采用3*delta
# 原则进行异常值识别，此处我们也可以简单计算下异常值范围： 这里标签是人工的很可能有特殊的含义
statistics = train['target'].describe()
logger.info(statistics.loc['mean'] - 3 * statistics.loc['std'])  # -11.945136285536142
logger.info(statistics.loc['mean'] + 3 * statistics.loc['std'])  # 11.157863687380166

# 规律一致性分析
# 所谓规律一致性，指的是需要对训练集和测试集特征数据的分布进行简单比对，以“确定”两组数据是否诞生于同一个总体
# 即两组数据是否都遵循着背后总体的规律，即两组数据是否存在着规律一致性。

# 特征列名
features = ['first_active_month', 'feature_1', 'feature_2', 'feature_3']

# 训练集/测试集样本总数
train_count = train.shape[0]
test_count = test.shape[0]

train['first_active_month'].value_counts().sort_index() / train_count

(train['first_active_month'].value_counts().sort_index() / train_count).plot()
plt.show()

for feature in features:
    (train[feature].value_counts().sort_index() / train_count).plot()
    (test[feature].value_counts().sort_index() / test_count).plot()
    plt.legend(['train', 'test'])
    plt.xlabel(feature)
    plt.ylabel('ratio')
    plt.show()


# 多变量联合分布
# 所谓联合概率分布，指的是将离散变量两两组合，然后查看这个新变量的相对占比分布。
# 例如特征1有0/1两个取值水平，特征2有A/B两个取值水平，则联合分布中就将存在0A、0B、1A、1B四种不同取值水平，
# 然后进一步查看这四种不同取值水平出现的分布情况

def combine_feature(df):
    cols = df.columns
    feature1 = df[cols[0]].astype(str).values.tolist()
    feature2 = df[cols[1]].astype(str).values.tolist()
    return pd.Series([feature1[i] + '&' + feature2[i] for i in range(df.shape[0])])


# 选取两个特征
cols = [features[0], features[1]]
logger.info(cols)  # ['first_active_month', 'feature_1']
train_com = combine_feature(train[cols])
# logger.info(train_com)  # 查看合并后结果 201916    2017-07&3
train_dis = train_com.value_counts().sort_index() / train_count
# logger.info(train_dis)  # 计算占比分布
# 对测试集进行相同的操作
test_dis = combine_feature(test[cols]).value_counts().sort_index() / test_count
logger.info(test_dis)

# 比较两者的分布
index_dis = pd.Series(train_dis.index.tolist() + test_dis.index.tolist()).drop_duplicates().sort_values()

# 对缺失值填补为0
(index_dis.map(train_dis).fillna(0)).plot()
(index_dis.map(test_dis).fillna(0)).plot()

# 绘图
plt.legend(['train', 'test'])
plt.xlabel('&'.join(cols))
plt.ylabel('ratio')
plt.show()

n = len(features)
for i in range(n - 1):
    for j in range(i + 1, n):
        cols = [features[i], features[j]]
        print(cols)
        train_dis = combine_feature(train[cols]).value_counts().sort_index() / train_count
        test_dis = combine_feature(test[cols]).value_counts().sort_index() / test_count
        index_dis = pd.Series(train_dis.index.tolist() + test_dis.index.tolist()).drop_duplicates().sort_values()
        (index_dis.map(train_dis).fillna(0)).plot()
        (index_dis.map(train_dis).fillna(0)).plot()
        plt.legend(['train', 'test'])
        plt.xlabel('&'.join(cols))
        plt.ylabel('ratio')
        plt.show()

# 联合分布规律差不多一致可以训练
# 1.如果分布非常一致，则说明所有特征均取自同一整体，训练集和测试集规律拥有较高一致性，模型效果上限较高，建模过程中应该更加依靠特征工程方法和模型建模技巧提高最终预测效果；
# 2.如果分布不太一致，则说明训练集和测试集规律不太一致，此时模型预测效果上限会受此影响而被限制，并且模型大概率容易过拟合，在实际建模过程中可以多考虑使用交叉验证等方式防止过拟合，并且需要注重除了通用特征工程和建模方法外的trick的使用
