# encoding:utf-8
"""
# 1 typing
python3中的typing模块提高代码健壮性

很多人在写完代码一段时间后回过头看代码，很可能忘记了自己写的函数需要传什么参数，返回什么类型的结果，
就不得不去阅读代码的具体内容，降低了阅读的速度，加上Python本身就是一门弱类型的语言，这种现象就变得更加的严重，
而typing这个模块很好的解决了这个问题。

## 1.1 typing模块的作用：
类型检查，防止运行时出现参数和返回值类型不符合。
作为开发文档附加说明，方便使用者调用时传入和返回参数类型。
该模块加入后并不会影响程序的运行，不会报正式的错误，只有提醒。


typing常用的类型：
int,long,float: 整型,长整形,浮点型;
bool,str: 布尔型，字符串类型；
List, Tuple, Dict, Set:列表，元组，字典, 集合;
Iterable,Iterator:可迭代类型，迭代器类型；
Generator：生成器类型；

## 1.2 example:
```
from typing import List, Tuple, Dict
def add(a:int, string:str, f:float, b:bool) -> Tuple[List, Tuple, Dict, bool]:
    list1 = list(range(a))
    tup = (string, string, string)
    d = {"a":f}
    bl = b
    return list1, tup, d,bl
print(add(5,"hhhh", 2.3, False))
# 结果：([0, 1, 2, 3, 4], ('hhhh', 'hhhh', 'hhhh'), {'a': 2.3}, False)
```
说明：
在传入参数时通过“参数名:类型”的形式声明参数的类型；
返回结果通过"-> 结果类型"的形式声明结果的类型。
在调用的时候如果参数的类型不正确pycharm会有提醒，但不会影响程序的运行。
对于如list列表等，还可以规定得更加具体一些，如：“-> List[str]”,规定返回的是列表，并且元素是字符串。

"""
from feeder import VarFeeder

import pandas as pd
import numpy as np
import argparse
from typing import Union, Iterable, Tuple, Dict, List, Any, Collection
import numba
import re


import os


def read_file(file) -> pd.DataFrame:

    if os.path.exists(file):
        if file.endswith('.pkl'):
            df = pd.read_pickle(file).set_index('Page')
            df.columns = df.columns.astype('M8[D]')
            return df
        elif file.endswith('.csv'):
            df = pd.read_csv(file)
            df.to_pickle(file+'.pkl')
            df = df.set_index('Page')
            df.columns = df.columns.astype('M8[D]')
            return df
        else:
            print('the file is not correct ')
    else:
        print('file is not in zhe path :%s' %file )



@numba.jit(nopython=True)
def find_start_end(data:np.ndarray):
    n_pages, n_days = data.shape
    start_ids = np.full(shape=n_pages, fill_value=-1, dtype=np.int32)
    end_ids = np.full(shape=n_pages, fill_value=-1, dtype=np.int32)

    for page in range(n_pages):
        for day in range(n_days):
            if not np.isnan(data[page, day]) and data[page, day] >0:
                start_ids[page] = day
                break
        for day in range(n_days-1, -1, -1):
            if not np.isnan(data[page, day]) and data[page, day] >0:
                end_ids[page] = day
                break
    return start_ids, end_ids



def prepare_data(start, end, valid_threshold) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray,np.ndarray]:
    """ 1 数据预处理
    :param start:
    :param end:
    :param valid_threshold:
    :return:


    """
    # 1 读取原始数据
    path = os.path.join('/home/zt/Documents/Data/kaggle_web_traffic', 'all.pkl')
    if os.path.exists(path=path):
        df = pd.read_pickle(path=path)
    else:
        # 读取train_2
        df = read_file('/home/zt/Documents/Data/kaggle_web_traffic/train_2.csv')
        scraped = read_file('/home/zt/Documents/Data/kaggle_web_traffic/2017-08-15_2017-09-11.csv')
        df[pd.Timestamp('2017-09-10')] = scraped['2017-09-10']
        df[pd.Timestamp('2017-09-11')] = scraped['2017-09-11']

        df = df.sort_index()
        # Cache result
        df.to_pickle(path)

    # 2 用户GoogleAnalitycsRoman的数据很差，属于拥挤路段，所以删除
    bad_roman = df.index.str.startswith("User:GoogleAnalitycsRoman")
    df = df[~bad_roman]
    if start and end:
        df = df.loc[:, start:end]
    elif end:
        df = df.loc[:, :end]

    # 3
    starts, ends = find_start_end(df.values)
    page_mask = (ends - starts) / df.shape[1] < valid_threshold
    print("Masked %d pages from %d" % (page_mask.sum(), len(df)))
    df = df[~page_mask]
    nans = pd.isnull(df)

    return np.log1p(df.fillna(0)), nans, starts[~page_mask], ends[~page_mask]



def uniq_page_map(pages:Collection):
    import re
    result = np.full(shape=[len(pages), 4], fill_value=-1, dtype=np.int32)
    pat = re.compile('(.+(?:(?:wikipedia\.org)|(?:commons\.wikimedia\.org)|(?:www\.mediawiki\.org)))_([a-z_-]+?)')
    prev_page = None
    num_page = -1
    agents = {'all-access_spider': 0, 'desktop_all-agents': 1, 'mobile-web_all-agents': 2, 'all-access_all-agents': 3}
    for i , entity in enumerate(pages):
        match = pat.fullmatch(string=entity)
        assert match
        page = match.group(1)
        agent = match.group(2)
        if page != prev_page:
            prev_page = page
            num_page += 1
        result[num_page, agents[agent]] = i
    return result[: num_page+1]

@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0

@numba.jit(nopython=True)
def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len/lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_365 = single_autocorr(series, lag)
            c_364 = single_autocorr(series, lag-1)
            c_366 = single_autocorr(series, lag+1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.NaN
    return corr #, support

def normalize(values: np.ndarray):
    return (values - values.mean()) / np.std(values)

term_pat = re.compile('(.+?):(.+)')
pat = re.compile(
    '(.+)_([a-z][a-z]\.)?((?:wikipedia\.org)|(?:commons\.wikimedia\.org)|(?:www\.mediawiki\.org))_([a-z_-]+?)$')

def extract(source) -> pd.DataFrame:
    """
    Extracts features from url. Features: agent, site, country, term, marker
    :param source: urls
    :return: DataFrame, one column per feature
    """
    if isinstance(source, pd.Series):
        source = source.values
    agents = np.full_like(source, np.NaN)
    sites = np.full_like(source, np.NaN)
    countries = np.full_like(source, np.NaN)
    terms = np.full_like(source, np.NaN)
    markers = np.full_like(source, np.NaN)

    for i in range(len(source)):
        l = source[i]
        match = pat.fullmatch(l)
        assert match, "Non-matched string %s" % l
        term = match.group(1)
        country = match.group(2)
        if country:
            countries[i] = country[:-1]
        site = match.group(3)
        sites[i] = site
        agents[i] = match.group(4)
        if site != 'wikipedia.org':
            term_match = term_pat.match(term)
            if term_match:
                markers[i] = term_match.group(1)
                term = term_match.group(2)
        terms[i] = term

    return pd.DataFrame({
        'agent': agents,
        'site': sites,
        'country': countries,
        'term': terms,
        'marker': markers,
        'page': source
    })

def make_page_features(pages: np.ndarray) -> pd.DataFrame:
    """
    Calculates page features (site, country, agent, etc) from urls
    :param pages: Source urls
    :return: DataFrame with features as columns and urls as index
    """
    tagged = extract(pages).set_index('page')
    # Drop useless features
    features: pd.DataFrame = tagged.drop(['term', 'marker'], axis=1)
    return features

def encode_page_features(df) -> Dict[str, pd.DataFrame]:
    """
    Applies one-hot encoding to page features and normalises result
    :param df: page features DataFrame (one column per feature)
    :return: dictionary feature_name:encoded_values. Encoded values is [n_pages,n_values] array
    """
    def encode(column) -> pd.DataFrame:
        one_hot = pd.get_dummies(df[column], drop_first=False)
        # noinspection PyUnresolvedReferences
        return (one_hot - one_hot.mean()) / one_hot.std()

    return {str(column): encode(column) for column in df}

def lag_indexes(begin, end) -> List[pd.Series]:
    """
    Calculates indexes for 3, 6, 9, 12 months backward lag for the given date range
    :param begin: start of date range
    :param end: end of date range
    :return: List of 4 Series, one for each lag. For each Series, index is date in range(begin, end), value is an index
     of target (lagged) date in a same Series. If target date is out of (begin,end) range, index is -1
    """
    dr = pd.date_range(begin, end)
    # key is date, value is day index
    base_index = pd.Series(np.arange(0, len(dr)), index=dr)

    def lag(offset):
        dates = dr - offset
        return pd.Series(data=base_index.loc[dates].fillna(-1).astype(np.int16).values, index=dr)

    return [lag(pd.DateOffset(months=m)) for m in (3, 6, 9, 12)]

def run():
    # 外部提交参数
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('data_dir')
    parser.add_argument('--valid_threshold', default=0.0, type=float, help="Series minimal length threshold (pct of data length)")
    parser.add_argument('--add_days', default=64, type=int, help="Add N days in a future for prediction")
    parser.add_argument('--start', help="Effective start date. Data before the start is dropped")
    parser.add_argument('--end', help="Effective end date. Data past the end is dropped")
    parser.add_argument('--corr_backoffset', default=0, type=int, help='Offset for correlation calculation')
    args = parser.parse_args()

    df, nans, starts, ends = prepare_data(args.start, args.end, args.valid_threshold)

    data_start, data_end = df.columns[0], df.columns[-1]

    # 往前创建几天的数据，用来预测
    features_end = data_end + pd.Timedelta(args.add_days, unit='D')
    print(f"start: {data_start}, end:{data_end}, features_end:{features_end}")

    #
    assert df.index.is_monotonic_increasing  # 判断索引是不是增长的
    page_map = uniq_page_map(df.index.values)

    # Yearly(annual) autocorrelation
    raw_year_autocorr = batch_autocorr(df.values, 365, starts, ends, 1.5, args.corr_backoffset)
    year_unknown_pct = np.sum(np.isnan(raw_year_autocorr))/len(raw_year_autocorr)  # type: float

    # Quarterly autocorrelation
    raw_quarter_autocorr = batch_autocorr(df.values, int(round(365.25/4)), starts, ends, 2, args.corr_backoffset)
    quarter_unknown_pct = np.sum(np.isnan(raw_quarter_autocorr)) / len(raw_quarter_autocorr)  # type: float

    print("Percent of undefined autocorr = yearly:%.3f, quarterly:%.3f" % (year_unknown_pct, quarter_unknown_pct))

    # Normalise all the things
    year_autocorr = normalize(np.nan_to_num(raw_year_autocorr))
    quarter_autocorr = normalize(np.nan_to_num(raw_quarter_autocorr))

    # Calculate and encode page features
    page_features = make_page_features(df.index.values)
    encoded_page_features = encode_page_features(page_features)

    # Make time-dependent features
    features_days = pd.date_range(data_start, features_end)
    #dow = normalize(features_days.dayofweek.values)
    week_period = 7 / (2 * np.pi)
    dow_norm = features_days.dayofweek.values / week_period
    dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)

    # Assemble indices for quarterly lagged data
    lagged_ix = np.stack(lag_indexes(data_start, features_end), axis=-1)

    page_popularity = df.median(axis=1)
    page_popularity = (page_popularity - page_popularity.mean()) / page_popularity.std()

    # Put NaNs back
    df[nans] = np.NaN

    # Assemble final output
    tensors = dict(
        hits=df,
        lagged_ix=lagged_ix,
        page_map=page_map,
        page_ix=df.index.values,
        pf_agent=encoded_page_features['agent'],
        pf_country=encoded_page_features['country'],
        pf_site=encoded_page_features['site'],
        page_popularity=page_popularity,
        year_autocorr=year_autocorr,
        quarter_autocorr=quarter_autocorr,
        dow=dow,
    )
    plain = dict(
        features_days=len(features_days),
        data_days=len(df.columns),
        n_pages=len(df),
        data_start=data_start,
        data_end=data_end,
        features_end=features_end

    )

    # Store data to the disk
    VarFeeder(args.data_dir, tensors, plain)


if __name__ == '__main__':
    run()
    """
    python make_feature.py /home/zt/Documents/Data/kaggle_web_traffic/data/var --add_days=63
    """
































