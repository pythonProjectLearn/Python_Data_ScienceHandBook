# encoding:utf-8
"""
策略1：选择当天阳包阴的股票：
      条件1：今天的收盘价高于昨天的开盘价；
      条件2：今天的开盘价低于昨天的收盘价
      条件3：昨今两天的数据满足昨天下跌超过2%
      条件4：今天的涨幅大于9.8

"""
from postgresql import postgreOperation
import pandas as pd
import numpy as np
import argparse
import datetime


if __name__=='__main__':
    parse = argparse.ArgumentParser(description='stock strategy one')
    parse.add_argument('--save_file', require=True, type=str, help='save file path')

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    yesterday_d = datetime.datetime.now() - datetime.timedelta(days=1)
    yesterday = yesterday_d.strftime('%Y-%m-%d')

    config = {'host': 'localhost', 'port': 5432, 'user': 'postgres', 'password': 'postgres', 'dbname': 'zhoutao'}
    postgre = postgreOperation(config)

    # 筛掉9999
    stocks_today = postgre.read_sql("select code,name, close_price_yesterday,open_price, high_price, low_price, last_price, up_down_60day,amplitude, update_time from stocks_a where open_price<>9999 and to_char(update_time, 'YYYY-mm-dd')='{}'".format(today))
    stocks_yesterday = postgre.read_sql("select name, close_price_yesterday,open_price, high_price, low_price, last_price, up_down_60day,amplitude, update_time from stocks_a where open_price<>9999 and to_char(update_time, 'YYYY-mm-dd')='{}'".format(yesterday))
    stocks_today_pd = pd.DataFrame(np.array(stocks_today),columns=["name", "close_price_yesterday","open_price", "high_price", "low_price", "last_price", "up_down_60day","amplitude","update_time"])
    stocks_yesterday_pd = pd.DataFrame(np.array(stocks_yesterday),
                                   columns=["name", "close_price_yesterday_0", "open_price_0", "high_price_0", "low_price_0", "last_price_0", "up_down_60day_0", "amplitude_0","update_time_0"])
    today_yesterday = pd.merge(stocks_today_pd, stocks_yesterday_pd, how='inner',on='name')

    # 阳包阴
    strategy_one = today_yesterday[(today_yesterday['last_price']>today_yesterday["open_price_0"]) & \
                    (today_yesterday['open_price']<today_yesterday["close_price_yesterday"]) & \
                    (today_yesterday['up_down_60day'] - today_yesterday['up_down_60day_0']<-2.) & \
                    (today_yesterday['amplitude'] > 9.8)
        ]
    try:
        # 将筛选出来的股票插入到数据库中
        stocks_tuple = ','.join([tuple(i) for i in strategy_one[['code', 'name', 'update_time']].values])
        postgre.write_sql('insert into stocks_filter (code, name, update_time) values {}'.format(stocks_tuple))
    except Exception as e:
        print('can not find a stocks that satisfied the strategy of yang_wrap_yin ')

