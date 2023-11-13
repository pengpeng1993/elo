import pandas as pd
import logging
import numpy as np

# 创建logger对象
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# 创建FileHandler对象
fh = logging.FileHandler('./log/02_date_analysis.log', "w")
fh.setLevel(logging.DEBUG)
# 创建Formatter对象
formatter = logging.Formatter()
fh.setFormatter(formatter)
# 将FileHandler对象添加到Logger对象中
logger.addHandler(fh)

merchant = pd.read_csv('../data/primeval/merchants.csv', header=0)
pd.set_option('display.max_columns', None)  # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行
logger.info(merchant.head(5))
print(merchant.info())

# #merchant_id 商户id
# merchant_group_id 商户组id
# merchant_category_id 商户类别id
# subsector_id 商品种类群id
# numerical_1 匿名数值特征1
# numerical_2 匿名数值特征2
# category_1 匿名离散特征1
# most_recent_sales_range 上个活跃月份收入等级，有序分类变量A>B>...>E
# most_recent_purchases_range 上个活跃月份交易数量等级，有序分类变量A>B>...>E
# avg_sales_lag3/6/12 过去3、6、12个月的月平均收入除以上一个活跃月份的收入
# avg_purchases_lag3/6/12 过去3、6、12个月的月平均交易量除以上一个活跃月份的交易量
# active_months_lag3/6/12 过去3、6、12个月的活跃月份数量
# category_2 匿名离散特征2

logger.info((merchant.shape, merchant['merchant_id'].nunique()))  # 在一个商户有多条记录

# 对比商户数据特征是否和数据字典中特征
df = pd.read_excel('../data/primeval/Data_Dictionary.xlsx', header=2, sheet_name='merchant')
logger.info(pd.Series(merchant.columns.tolist()).sort_values().values == pd.Series(
    [va[0] for va in df.values]).sort_values().values)

logger.info(merchant.isnull().sum())  # 第二个匿名分类变量存在较多缺失值  avg_sales_lag3/6/12缺失值数量一致，则很有可能是存在13个商户同时确实了这三方面信息

category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
                 'subsector_id', 'category_1',
                 'most_recent_sales_range', 'most_recent_purchases_range',
                 'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
                'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']

# 检验特征是否划分完全
logger.info(len(category_cols) + len(numeric_cols) == merchant.shape[1])

logger.info(merchant[category_cols].nunique())
logger.info(merchant[category_cols].dtypes)

# 查看离散变量的缺失值情况
logger.info(merchant[category_cols].isnull().sum())

logger.info(merchant['category_2'].unique())
merchant['category_2'] = merchant['category_2'].fillna(-1)


# 离散变量编码
# 变量类型应该是有三类，分别是连续性变量、名义型变量以及有序变量。连续变量较好理解，
# 所谓名义变量，指的是没有数值大小意义的分类变量，例如用1表示女、0表示男，0、1只是作为性别的指代，而没有1>0的含义。 独热编码
# 而所有有序变量，其也是离散型变量，但却有数值大小含义，如上述most_recent_purchases_range字段，销售等级中A>B>C>D>E，该离散变量的5个取值水平是有严格大小意义的，该变量就被称为有序变量。


# 字典编码函数
def change_object_cols(se):
    value = se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values


for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = change_object_cols(merchant[col])

# 连续变量的数据探索
logger.info(merchant[numeric_cols].dtypes)
logger.info(merchant[numeric_cols].isnull().sum())
logger.info(merchant[numeric_cols].describe())

# 据此我们发现连续型变量中存在部分缺失值，并且部分连续变量还存在无穷值inf，需要对其进行简单处理。
inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())

# 缺失值处理
# 不同于无穷值的处理，缺失值处理方法有很多。但该数据集缺失数据较少，33万条数据中只有13条连续特征缺失值，此处我们先简单采用均值进行填补处理，后续若有需要再进行优化处理。
for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())

logger.info(merchant[numeric_cols].describe())
