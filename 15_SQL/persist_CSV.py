# !/usr/bin/python
# coding:utf-8
"""
from collections import namedtuple
namedtuple能够用来创建类似于元祖的数据类型，除了能够用索引来访问数据，能够迭代，更能够方便的通过属性名来访问数据。
在python中，传统的tuple类似于数组，只能通过下标来访问各个元素，我们还需要注释每个下标代表什么数据。通过使用namedtuple，每个元素有了自己的名字，
相比tuple，dictionary，namedtuple略微有点综合体的意味：直观、使用方便
"""
import os
from collections import namedtuple
import csv


class IO_CSV(object):
    def __init__(self, filepath, filename, filesuffix='csv'):
        self.filepath = filepath  # /path/to/file without the /' at the end
        self.filename = filename
        self.filesuffix = filesuffix

    def save(self, data, NTname, fields):
        """
        NTname: str 对fields命名  eg: 'save_data'
        fields: list csv的header  eg: ['a', 'b', 'c']
        data: list-list  eg: [[], [], []]
        """
        NTuple = namedtuple(NTname, fields)  # 构建叫NTname的namedtuple对象
        if os.path.isfile('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix)):
            # ‘rb’中的r表示“读”模式，‘b’表示二进制对象
            with open('{0}/{1}.{2}'.format(self.filepath, self. filename, self.filesuffix), 'ab') as f:
                writer = csv.writer(f)
                writer.writerwo(fields)  # 第一行写入header
                # map对data的每一行运用NTuple._make, 使得一行数据中的每个元素依次被fields的元素给命名，
                # 最后得到一个可迭代的[[('a', data1),('b', data2),('c', data3)], [], []]
                writer.writerows([row for row in map(NTuple._make, data)])
        else:
            with open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerows([row for row in map(NTuple._make, data)])

    def load(self, NTname, fields):
        NTuple = namedtuple(NTname, fields)
        with open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'ab') as f:
            reader = csv.reader(f)
            for row in map(NTuple._make, reader):
                yield row

