import logging
import pandas as pd

# 创建logger对象
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# 创建FileHandler对象
fh = logging.FileHandler('./log/03_date_analysis.log', "w")
fh.setLevel(logging.DEBUG)
# 创建Formatter对象
formatter = logging.Formatter()
fh.setFormatter(formatter)
# 将FileHandler对象添加到Logger对象中
logger.addHandler(fh)
pd.set_option('display.max_columns', None)  # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行

history_transaction = pd.read_csv('../data/primeval/historical_transactions.csv', header=0)
logger.info(history_transaction.head(5))
print(history_transaction.info())

# | 字段 | 解释 |
# | ------ | ------ |
# | card_id | 独一无二的信用卡标志 |
# | authorized_flag | 是否授权，Y/N |
# | city_id | 城市id，经过匿名处理 |
# | category_1 | 匿名特征，Y/N |
# | installments | 分期付款的次数 |
# | category_3 | 匿名类别特征，A/.../E |
# | merchant_category_id | 商户类别，匿名特征 |
# | merchant_id | 商户id |
# | month_lag	 | 距离2018年月的2月数差 |
# | purchase_amount | 标准化后的付款金额 |
# | purchase_date | 付款时间 |
# | category_2 | 匿名类别特征2 |
# | state_id | 州id，经过匿名处理 |
# | subsector_id | 商户类别特征 |

new_transaction = pd.read_csv('../data/primeval/new_merchant_transactions.csv', header=0)
merchant = pd.read_csv('../data/primeval/merchants.csv', header=0)
logger.info(new_transaction.head(5))
print(new_transaction.info())

# why 交易表商户id不能重复
duplicate_cols = []
for col in merchant.columns:
    if col in new_transaction.columns:
        duplicate_cols.append(col)
logger.info(duplicate_cols)

# 取出和商户数据表重复字段并去重
logger.info(new_transaction[duplicate_cols].drop_duplicates().shape) #(291242, 7)
logger.info(new_transaction['merchant_id'].nunique()) # 226129

numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
                 'subsector_id']
time_cols = ['purchase_date']

assert len(numeric_cols) + len(category_cols) + len(time_cols) == new_transaction.shape[1]

logger.info(new_transaction[category_cols].dtypes)
logger.info(new_transaction[category_cols].isnull().sum())


def change_object_cols(se):
    value = se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values


# 和此前的merchant处理类似，我们对其object类型对象进行字典编码（id除外），并对利用-1对缺失值进行填补：
for col in ['authorized_flag', 'category_1', 'category_3']:
    new_transaction[col] = change_object_cols(new_transaction[col].fillna(-1).astype(str))

new_transaction[category_cols] = new_transaction[category_cols].fillna(-1)

logger.info(new_transaction[category_cols].dtypes)
