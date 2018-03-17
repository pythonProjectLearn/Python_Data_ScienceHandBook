# coding:utf-8
import os,io
import json


class IO_Json(object):
    """
    #这两个函数可以看做是用来变换内存中的数据类型的,是数据类型的转变
    json.dumps()将Python中的list或者dict类型数据({'a': True,'b': 'Hello',}),编码成json类型的数据(即类似一条字符串'{'a': True,'b': 'Hello',}')
    json.loads()将json类型数据(注意txt文件也可以存储)，解码成python的类型数据

    # 下面两个函数是用来处理文件的,是文件的操作
    json.dump()是将json类型数据(由json.dumps得到)，保存到json文件中
    json.load()从文件中存储的json格式的数据读取出来(然后用json.loads将json格式数据解码成python格式数据)
    """
    def __init__(self, filepath, filename, filesuffix='json'):
        self.filepath = filepath  # /path/to/file without the /' at the end
        self.filename = filename        # FILE_NAME
        self.filesuffix = filesuffix
        # self.file_io = os.path.join(dir_name, .'.join((base_ filename, filename_suffix)))

    def save(self, data):
        """通过json.dumps()python的dict转化成json格式,然后通过json.dump()将json格式的数据写入json文件
        :param data: dict或list
        :return: 写入json文件
        """
        if os.path.isfile('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix)):
            # 存在json文件时,打开文件插入数据 在python3中不需要unicode这个函数
            with io.open('{0}/{1}.{2}'.format(self.filepath, self. filename, self.filesuffix), 'a', encoding='utf-8') as f:
                # f.write(unicode(json.dumps(data, ensure_ascii=False)))
                # 通过json.dump将json类型的数据，写入json文件中
                json.dump(unicode(json.dumps(data, ensure_ascii=False)), f)
        else:
            # 不存在json文件时，建立文件插入数据
            with io.open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'w', encoding='utf-8') as f:
                #f.write(unicode(json.dumps(data, ensure_ascii=False)))
                json.dump(unicode(json.dumps(data, ensure_ascii=False)), f)

    def load(self):
        """读取json文件(其实这个文件，也不一定非要是以.json为后缀的名字，txt文件也可以，只要里面存储的数据格式是json类型就好了)
        返回python格式的数据
        """
        with io.open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'r', encoding='utf-8') as f:
            # json.load()读取json格式数据, json.loads将json格式数据变成python格式数据
            return json.loads(json.load(f))


