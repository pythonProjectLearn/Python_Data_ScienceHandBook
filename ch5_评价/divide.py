# coding:utf-8
# 十折交叉验证，将数据集分割成10（份）桶数据
import random

def buckets(filename, bucketName, separator, classColumn):
    """
    目的：将原始数据分割成10份，并写入还有bucketName前缀的文件中
    filename：原始数据集的名称
    bucketName：被分成10个桶。每个桶的名字的前缀（要被写入的数据集的前缀名称）
    separator：分割每一列数据的符号，在pimaSmall.txt中，列数据是用，分割的
    classColumn：data数据集中有7列数据，第7列代表该行数据的类别，则classColumn=7
    (for ex., a tab or comma and classColumn is the column
    that indicates the class"""
    numberOfBuckets = 10    #要将数据分割成10份
    data = {}   # 数据集含有类别，用字典包装，key代表类别
    # 取得原始数据fielname文件到lines中
    with open(filename) as f:
        lines = f.readlines()
    # 遍历每一行。
    for line in lines:
        #如果每列的分割符不是\t, 则用\t替换
        if separator != '\t':
            line = line.replace(separator, '\t')
        # 得到每一行数据的类别
        category = line.split()[classColumn]
        data.setdefault(category, [])   #每个类别中用列表包装
        data[category].append(line)     #字典的key是唯一的，所以每一行数据用append添加line，那么每一行包含在一个列表中
    # 初始化这10个桶
    buckets = []
    #在buckets中装上10个小桶
    for i in range(numberOfBuckets):
        buckets.append([])       
    # 遍历所有类别
    for k in data.keys():
        #将每个类别data[k]中的列表数据打乱
        random.shuffle(data[k])
        bNum = 0
        # 遍历一个类别中所有列表（每个列表代表一行数据）
        for item in data[k]:
            #将所有行的数据分割成10份，装入buckets中
            buckets[bNum].append(item)
            bNum = (bNum + 1) % numberOfBuckets

    #将分割后的10个数据集分别写入 bucketName+bNub的名称的txt中
    for bNum in range(numberOfBuckets):
        f = open("%s-%02i" % (bucketName, bNum + 1), 'w')
        #遍历buckets中的每个桶中的数据
        for item in buckets[bNum]:
            f.write(item)
        f.close()

# example of how to use this code          
#buckets("pimaSmall.txt", 'pimaSmall',',',8)
