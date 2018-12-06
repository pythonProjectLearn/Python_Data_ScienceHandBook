# encoding:utf-8
import psycopg2
import logging
import pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.DEBUG,  # 日志水平，
                    # 每条日志记录的形式
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',  # 每条日志记录生成的时间
                    filename=os.path.join(current_dir, 'postgresql.log'),  # 日志的名称
                    filemode='w'  # 日志覆盖写入
                    )

# 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class Postgresql(object):
    def __init__(self, config):
        try:
            self.conn = psycopg2.connect(host=config['host'],
                                         port=config['port'],
                                         user=config['user'],
                                         password=config['password'],
                                         database=config['database'])
        except psycopg2.DatabaseError as e:
            print('连接postgresql失败', e)

    def selectSql(self, sql):
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                return result
            except psycopg2.Error as e:
                self.conn.rollback()
                print('select事务失败', e)

    def insertSql(self, sql):
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                self.conn.commit()
                self.conn.close()
            except psycopg2.Error as e:
                self.conn.rollback()
                print("insert 事务失败", e)

    def updateSql(self, sql):
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                self.conn.commit()
                self.conn.close()
            except psycopg2.Error as e:
                self.conn.rollback()
                print("insert 事务失败", e)

    def delSql(self, sql):
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                self.conn.commit()
                self.conn.close()
            except psycopg2.Error as e:
                self.conn.rollback()
                print("insert 事务失败", e)



if __name__ == "__main__":
    config={'host':'localhost', 'port':'5432', 'user':'postgres', 'password':'postgres', 'database':'zhoutao'}
    ps = Postgresql(config=config)
    result = ps.selectSql("select * from stocks_a where code='600117'")