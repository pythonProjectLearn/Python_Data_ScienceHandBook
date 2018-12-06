# coding:utf-8

from pymongo import MongoClient as MCli

class IO_Mongo(object):
    conn = {'host': 'localhost', 'ip': '27017'}
    def __init__(self, db='twtr_db', coll='twtr_coll', **conn):
        """
        :param db: 数据库
        :param coll: collection 相当于数据表, collection当中的一条数据称为document(文档)
        :param conn: 可以看做是游标
        """
        self.client = MCli(**conn)
        self.db = self.client[db]
        self.coll = self.db[coll]

    def save(self, data):
        return self.coll.insert(data)

    def load(self, return_cursor=False, criteria=None, projection=None):
        """
        :param return_cursor: 返回的结果
        :param criteria: 查询的条件
        :param projection:
        :return: 将所有返回的结果装在list中
        """
        if criteria is None:
            criteria = {}
        if projection is None:
            cursor = self.coll.find(criteria)
        else:
            cursor = self.coll.find(criteria, projection)
        if return_cursor:
            return cursor
        else:
            return [item for item in cursor]
