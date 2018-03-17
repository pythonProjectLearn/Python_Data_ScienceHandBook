#  coding:utf-8
# 
#  Nearest Neighbor Classifier for Pima dataset
#
#
#  Code file for the book Programmer's Guide to Data Mining
#  http://guidetodatamining.com
#
#  Ron Zacharski
#
import heapq
import random

class Classifier:
    def __init__(self, bucketPrefix, testBucketNumber, dataFormat, k):
        """
        bucketPrefix：是原始数据集mpgData
        testBucketNumber：第几个测试数据集，例如一共有10个桶的数据，要测试第3个数据集，则它为3
        dataFormat：是一个如何解释数据中每列的字符串，如"num	num	num	num	num	num	num	num	class"
        k：10折交叉验证，则它为10
        """
        self.medianAndDeviation = []  #包装每一列的中位数 和 绝对值标准差
        self.k = k

        #将dataFormat转化成['num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'class']
        self.format = dataFormat.strip().split('\t')
        self.data = []
        # for each of the buckets numbered 1 through 10:
        for i in range(1, 11):
            #不为testBucketNumber的数据集是训练集
            if i != testBucketNumber:
                filename = "%s-%02i" % (bucketPrefix, i)
                f = open(filename)
                lines = f.readlines()
                f.close()
                #遍历所有行的数据，每一行的第一个数据是，该行数据的类别，所以不取
                for line in lines[1:]:
                    #将每一行数据，分成[col1, col2, col3]一列一列的数据
                    fields = line.strip().split('\t')
                    ignore = []
                    vector = []
                    classification = []
                    #遍历一行中的每个数据
                    for i in range(len(fields)):
                        if self.format[i] == 'num':
                            #当该列数据是num列时，放入vector中
                            vector.append(float(fields[i]))
                        elif self.format[i] == 'comment':
                            #当该列数据是comment列时，放入ignore中
                            ignore.append(fields[i])
                        elif self.format[i] == 'class':
                            #当该列数据是class列时，放入classification中
                            classification = fields[i]
                    self.data.append((classification, vector, ignore))
        # 将data转化为list列表
        self.rawData = list(self.data)
        # 得到每行向量的长度
        self.vlen = len(self.data[0][1])
        # 标准化data数据
        for i in range(self.vlen):
            self.normalizeColumn(i)
        

        
    
    ##################################################
    def getMedian(self, alist):
        """
       目的：得到alist向量的中位数
        alist：一个向量（列表）
        """
        if alist == []:
            return []
        blist = sorted(alist)  #对向量alist排序，得到blist
        length = len(alist)    #该列的长度
        if length % 2 == 1:
            # 当blist向量的长度为奇数时，返回blist的中位数
            return blist[int(((length + 1) / 2) -  1)]
        else:
            # 当blist是偶数时，返回blist中间两个数的平均数
            v1 = blist[int(length / 2)]
            v2 =blist[(int(length / 2) - 1)]
            return (v1 + v2) / 2.0
        

    def getAbsoluteStandardDeviation(self, alist, median):
        """
        目的：采用中位数，绝对值标准化alist向量（得到类似于标准差的值）
        alist：一个向量
        median：alist向量的中位数
        """
        sum = 0
        for item in alist:
            sum += abs(item - median)  #可以用map函数
        return sum / len(alist)


    def normalizeColumn(self, columnNumber):
       """
       目的：给定数据集data的第几列（即columnNumber），标准化该列的数据
       columnNumber：数据集的第几列
       结果：两个结果，1、得到getAbsoluteStandardDeviation列表，它包含所有列的中位数和绝对化标准差
             2、v
       """
       #data第0行数是列名，所以从v[1]开始，即从第一行开始遍历， col为data的第columnNumber列数
       col = [v[1][columnNumber] for v in self.data]
       median = self.getMedian(col)  #得到列col的中位数
       asd = self.getAbsoluteStandardDeviation(col, median) # 绝对化标准差的值
       #将一个列的中位数和绝对化标准差，这两个数放入medianAndDeviation中
       self.medianAndDeviation.append((median, asd))
       #标准化columnNumber列的数据
       for v in self.data:
           v[1][columnNumber] = (v[1][columnNumber] - median) / asd


    def normalizeVector(self, v):
        """
        目的：标准化向量v
        v：是一个行向量， 代表data的一行数据
        结果：一行的标准化数据
        """
        vector = list(v)
        for i in range(len(vector)):
            (median, asd) = self.medianAndDeviation[i]
            vector[i] = (vector[i] - median) / asd
        return vector

    ##################################################

    def testBucket(self, bucketPrefix, bucketNumber):
        """Evaluate the classifier with data from the file
        bucketPrefix-bucketNumber"""

        #读取10个桶的数据
        filename = "%s-%02i" % (bucketPrefix, bucketNumber)
        f = open(filename)
        lines = f.readlines()
        totals = {}
        f.close()
        for line in lines:
            data = line.strip().split('\t')
            vector = []
            classInColumn = -1   #默认data最后一列是数据的类型
            for i in range(len(self.format)):
                  if self.format[i] == 'num':
                      #非类别的数据集为num，放入vector中
                      vector.append(float(data[i]))
                  elif self.format[i] == 'class':
                      #指示类别数据是滴i列
                      classInColumn = i
            theRealClass = data[classInColumn]
            #print("REAL ", theRealClass)
            classifiedAs = self.classify(vector)
            totals.setdefault(theRealClass, {})
            totals[theRealClass].setdefault(classifiedAs, 0)
            totals[theRealClass][classifiedAs] += 1
        return totals



    def manhattan(self, vector1, vector2):
        """曼哈顿距离"""
        return sum(map(lambda v1, v2: abs(v1 - v2), vector1, vector2))


    def nearestNeighbor(self, itemVector):
        """结果：返回离itemVector最近的一个数据集"""
        return min([ (self.manhattan(itemVector, item[1]), item)
                     for item in self.data])
    
    def knn(self, itemVector):
        """
        结果：返回离itemVector最近的k个数据集
        """
        # heapq.nsmallest(n, iterable, key=None) 将iterable按照从小到大排列，取前n个最小的值
        neighbors = heapq.nsmallest(self.k,
                                   [(self.manhattan(itemVector, item[1]), item)
                                     for item in self.data]
                                    )
        # each neighbor gets a vote
        results = {}
        #遍历近邻的数据集，近邻数最多的类别，就是itemVector的预测类别
        for neighbor in neighbors: 
            theClass = neighbor[1][0]  #每个近邻的数据第1行第0列是类别
            results.setdefault(theClass, 0) #类别默认为0，类别每增加1，result加1
            results[theClass] += 1
        #将类别的个数，按从大到小排列，i[1]该类的数量，i[0]类别
        resultList = sorted([(i[1], i[0]) for i in results.items()], reverse=True)
        #得到排名第一位的类别
        maxVotes = resultList[0][0]
        possibleAnswers = [i[1] for i in resultList if i[0] == maxVotes]
        # 随机选择一个类别，得到最大的向量
        answer = random.choice(possibleAnswers)
        return( answer)
    
    def classify(self, itemVector):
        """Return class we think item Vector is in"""
        # 首先将itemVector标准化，然后使用knn得到k个近邻
        return(self.knn(self.normalizeVector(itemVector)))                             
 

 #10折交叉验证的执行流程
def tenfold(bucketPrefix, dataFormat, k):
    """
    目的：构造10个分类器，每个分类器利用9个桶中的数据进行训练，其余数据用于测试
    结果：打印出交叉验证的结果
    """
    results = {}
    for i in range(1, 11):
        c = Classifier(bucketPrefix, i, dataFormat, k)  #实例化数据集
        t = c.testBucket(bucketPrefix, i)
        for (key, value) in t.items():
            results.setdefault(key, {})
            for (ckey, cvalue) in value.items():
                results[key].setdefault(ckey, 0)
                results[key][ckey] += cvalue
                
    # 打印结果
    categories = list(results.keys())
    categories.sort()
    print(   "\n       Classified as: ")
    header =    "        "
    subheader = "      +"
    for category in categories:
        header += "% 2s   " % category
        subheader += "-----+"
    print (header)
    print (subheader)
    total = 0.0
    correct = 0.0
    for category in categories:
        row = " %s    |" % category 
        for c2 in categories:
            if c2 in results[category]:
                count = results[category][c2]
            else:
                count = 0
            row += " %3i |" % count
            total += count
            if c2 == category:
                correct += count
        print(row)
    print(subheader)
    print("\n%5.3f percent correct" %((correct * 100) / total))
    print("total of %i instances" % total)

#---------------------------------------------
import os
BASE_DIR = os.getcwd()
ch05path = os.path.join(BASE_DIR, 'data/ch06').replace('\\', '/')

print("SMALL DATA SET")
tenfold("pimaSmall/pimaSmall",
        "num	num	num	num	num	num	num	num	class", 3)

print("\n\nLARGE DATA SET")

tenfold("pima/pima",
        "num	num	num	num	num	num	num	num	class", 3)
