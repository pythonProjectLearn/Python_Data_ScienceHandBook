# coding:utf-8
from __future__ import print_function
import os, codecs, math

class BayesText:

    def __init__(self, trainingdir, stopwordlist):
        """This class implements a naive Bayes approach to text
        classification
        trainingdir:训练集数据路径
        stopwordlist ：停用词列表
        """
        self.vocabulary = {}
        self.prob = {}
        self.totals = {}
        self.stopwords = {}
        f = open(stopwordlist)
        #将停用词读入self.stopwords中
        for line in f:
            self.stopwords[line.strip()] = 1
        f.close()
        categories = os.listdir(trainingdir)
        #将路径中所有的文件数据读入self.categories中
        self.categories = [filename for filename in categories
                           if os.path.isdir(trainingdir + filename)]
        print("Counting ...")
        for category in self.categories:
            print('    ' + category)
            #self.train函数在后面定义了，prob[category]该类别中词汇的频数，totals[category]该类别词汇总频数
            (self.prob[category], self.totals[category]) = self.train(trainingdir, category)
        # I am going to eliminate any word in the vocabulary
        # that doesn't occur at least 3 times
        toDelete = []
        for word in self.vocabulary:
            if self.vocabulary[word] < 3:
                #找出单词量小于3的单词
                # mark word for deletion
                # can't delete now because you can't delete
                # from a list you are currently iterating over
                toDelete.append(word)
        # 删除单词量小3的单词
        for word in toDelete:
            del self.vocabulary[word]
        # 计算有多少个不重复的单词
        vocabLength = len(self.vocabulary)
        print("Computing probabilities:")
        for category in self.categories:
            print('    ' + category)
            #denominator（分母）
            denominator = self.totals[category] + vocabLength
            #遍历所有的词汇
            for word in self.vocabulary:
                #计算每个单词在category类别中的频数
                if word in self.prob[category]:
                    count = self.prob[category][word]
                else:
                    count = 1
                #通过频数，计算该单词在category类中的频率
                self.prob[category][word] = (float(count + 1)
                                             / denominator)
        print ("DONE TRAINING\n\n")
                    

    def train(self, trainingdir, category):
        """
        目的：计算一个种类(category)单词出现的次数,
        trainingdir：训练集的路径
        返回：每个词出现的个数counts，以及总的词汇量total
        """
        currentdir = trainingdir + category
        files = os.listdir(currentdir)
        counts = {}
        total = 0
        for file in files:
            #print(currentdir + '/' + file)
            f = codecs.open(currentdir + '/' + file, 'r', 'iso8859-1')
            for line in f:
                tokens = line.split()
                for token in tokens:
                    # get rid of punctuation and lowercase token
                    token = token.strip('\'".,?:-')  #strip()忽略这些字符
                    token = token.lower() #将所有字符小写
                    #排除空字符串和停用词
                    if token != '' and not token in self.stopwords:
                        self.vocabulary.setdefault(token, 0) #每个词的频数默认为0，vocabulary要存储所有文章的词汇表
                        self.vocabulary[token] += 1 #每出现一个词，其频数加1
                        counts.setdefault(token, 0)  #counts是某篇文章的词汇表
                        counts[token] += 1
                        total += 1
            f.close()
        return(counts, total)
                    
                    
    def classify(self, filename):
        results = {}
        for category in self.categories:
            results[category] = 0
        f = codecs.open(filename, 'r', 'iso8859-1')
        for line in f:
            tokens = line.split()
            for token in tokens:
                #print(token)
                token = token.strip('\'".,?:-').lower()
                if token in self.vocabulary:
                    for category in self.categories:
                        if self.prob[category][token] == 0:
                            print("%s %s" % (category, token))
                        results[category] += math.log(
                            self.prob[category][token])
        f.close()
        results = list(results.items())
        results.sort(key=lambda tuple: tuple[1], reverse = True)
        # for debugging I can change this to give me the entire list
        return results[0][0]

    def testCategory(self, directory, category):
        files = os.listdir(directory)
        total = 0
        correct = 0
        for file in files:
            total += 1
            result = self.classify(directory + file)
            if result == category:
                correct += 1
        return (correct, total)

    def test(self, testdir):
        """Test all files in the test directory--that directory is
        organized into subdirectories--each subdir is a classification
        category"""
        categories = os.listdir(testdir)
        #filter out files that are not directories
        categories = [filename for filename in categories if
                      os.path.isdir(testdir + filename)]
        correct = 0
        total = 0
        for category in categories:
            print(".", end="")
            (catCorrect, catTotal) = self.testCategory(
                testdir + category + '/', category)
            correct += catCorrect
            total += catTotal
        print("\n\nAccuracy is  %f%%  (%i test instances)" %
              ((float(correct) / total) * 100, total))
            
# change these to match your directory structure
baseDirectory = "/Users/raz/Dropbox/guide/data/20news-bydate/"
trainingDir = baseDirectory + "20news-bydate-train/"
testDir = baseDirectory + "20news-bydate-test/"


stoplistfile = "/Users/raz/Downloads/20news-bydate/stopwords0.txt"
print("Reg stoplist 0 ")
bT = BayesText(trainingDir, baseDirectory + "stopwords0.txt")
print("Running Test ...")
bT.test(testDir)

print("\n\nReg stoplist 25 ")
bT = BayesText(trainingDir, baseDirectory + "stopwords25.txt")
print("Running Test ...")
bT.test(testDir)

print("\n\nReg stoplist 174 ")
bT = BayesText(trainingDir, baseDirectory + "stopwords174.txt")
print("Running Test ...")
bT.test(testDir)
