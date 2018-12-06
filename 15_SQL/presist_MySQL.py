# coding:utf-8
"""
class Kls(object):
    def __init__(self, data):
        self.data = data

    def printd(self):
        print self.data

    @staticmethod
    def checkind():
        return (IND == 'ON')

    def do_reset(self):
        if self.checkind(): # 被实例化方法调用checkind
            print('Reset done for:', self.data)

    @classmethod
    def cmethod(cls, **kwargs):
        print "class", kwargs


self @classmethod @statstic 之间的区别:
self: self参数指向当前实例自身，
    它的方法即可以被，实例ik.printd()调用，也可以被类Kls.printd()调用

@classmethod:
如果现在我们想写一些仅仅与类交互而不是和实例交互的方法,那么使用@classmethod
实例方法的第一个参数是self, classmethod是类方法，类方法的第一个参数cls
在实例化类的时候，会先执行类方法classmethod，然后再执行实例方法，
所以mySqlSetting类方法得到dbpool会在IO_MySQL先执行，然后被__init__调用，然后再是实例化方法例如do_upsert
   如果是实例：先会调用类方法，再类里面运行之后，再执行实例
   如果是类： 会直接调用类方法

@staticmethod
经常有一些跟类有关系的功能但在运行时又不需要实例和类参与的情况下,需要用到静态方法
它不能被实例或类调用
   不能被实例调用：即不能使用ik.checfind()
   可以被类调用：Kls.checkfind()
"""
# Twisted 是一个异步网络框架
# wisted.enterprise.adbapi为此产生，它是DB-API 2.0 API的非阻塞接口，可以访问各种关系数据库

import hashlib
from datetime import datetime
from twisted.enterprise import adbapi
import logging

import pymysql


class MySQLPipeline(object):
    def __init__(self, dbpool):
        self.dbpool = dbpool

    @classmethod
    def from_settings(cls, settings):
        """
        from twisted.enterprise import adbapi是在异步框架twisted中调用各种数据库的api
        从而构建数据库连接池dbpool, 它会创建一个 ConnectionPool 对象来链接某个关系型数据库
        dbpool = adbapi.ConnectionPool("dbmodule", 'mydb', 'andrew', 'password')
        第一个参数是调用数据库的模块，例如MySQLdb pymysql
        """
        kwargs = dict(
            host=settings['MYSQL_HOST'],
            db=settings['MYSQL_DBNAME'],
            user=settings['MYSQL_USER'],
            passwd=settings['MYSQL_PASSWD'],
            cursorclass=pymysql.cursors.DictCursor,  # 设置游标类型为字典类型
            charset='utf8',
            use_unicode=True,
        )
        dbpool = adbapi.ConnectionPool('pymysql', **kwargs)
        return cls(dbpool)  # 返回给类


    def process_item(self, item, spider):
        """它是这个类中，类对象实例化后唯一执行的方法， 其他方法都被包含在这里"""
        if spider.name != 'website':
            d = self.dbpool.runInteraction(self._do_upsert, item, spider)
            d.addBoth(lambda _: item)
            return d
        logging.debug('processing website: %r' % (item))
        d = self.dbpool.runInteraction(self._do_upsert, item, spider)
        d.addErrback(self._handle_error, item, spider)  # 数据库连接池链接对象d，增加错误回调函数self._handle_error
        d.addBoth(lambda _: item)  # 线程池链接对象d,增加两个回调函数，一个是错误信息(errback)的回调，一个是正常信息的(callback)
        return d

    def _do_upsert(self, conn, item, spider):
        """上面的self.dbpool.runInteraction()在建立数据库链接池的同时，会创建游标对象，
        在给游标对象命名时，只需要在游标的操作方法_do_upsert()里的第一个参数命名游标就可以

        """
        guid = self._get_guid(item)
        now = datetime.utcnow().replace(microsecond=0).isoformat(' ') # '2017-01-15 02:37:32'

        # 首先通过md5加密的唯一值，判断是否已经存在mysql当中(md5加密后的数字只有被修改后加密数字才会变动)，
        # 如果guid已经存在了，则conn会停留在这一行，conn.fetchone()[0]则有值，否则ret为空值
        conn.execute("""SELECT EXISTS(
            SELECT 1 FROM website WHERE guid = %s
        )""", (guid, ))
        ret = conn.fetchone()[0]  # 用pymysql语法，执行游标操作获取单条数据

        # 如果已经存在数据则更新， 否则插入新数据
        if ret:
            conn.execute("""
                UPDATE website
                SET name=%s, description=%s, url=%s, updated=%s
                WHERE guid=%s
            """, (item['name'], item['description'], item['url'], now, guid))
            spider.log("Item updated in db: %s %r" % (guid, item))
        else:
            conn.execute("""
                INSERT INTO website (guid, name, description, url, updated)
                VALUES (%s, %s, %s, %s, %s)
            """, (guid, item['name'], item['description'], item['url'], now))
            spider.log("Item stored in db: %s %r" % (guid, item))

    def _handle_error(self, failure, item, spider):
        """将错误信息记录到日志中"""
        logging.error(failure)

    def _get_guid(self, item):
        """用appId经过md5加密成唯一值,方便数据更新时哈希匹配"""
        return hashlib.md5(item['appId']).hexdigest()



