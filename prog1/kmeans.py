import random
import pandas as pd
import numpy as np
from sklearn import preprocessing


#随机初始化中心点
def kMeansInitCentroids(X, k):
    #从X的数据中随机取k个作为中心点
    index=np.random.randint(0,len(X-1),k)
    return X[index]

#计算数据点到中心点的距离，并判断该数据点属于哪个中心点
def findClosestCentroids(X, centroids):
    #idx中数据表明对应X的数据是属于哪一个中心点的，创建数组idx
    idx = np.zeros(len(X))
#TODO: 补充代码，返回每个数据点属于的聚类中心
    


    return idx
#重新计算中心点位置
def computeCentroids(X, idx):
#TODO:补充代码重新计算中心点位置，返回新的中心点位置
    return centroids
def k_means(X, k, max_iters):
#TODO: 补充代码，实现k均值算法
    



    
    return idx,centroids

def savaData(filePath, data):
    '''
    用于保存输出结果到指定路径下
    :param filePath: 保存结果的目的文件路径
    :param data: 结果数据
    :return:
    '''
    file = open(filePath, 'w+', encoding='utf-8')  # 注意规定编码格式
    file.write(str(data))  # 写入结果数据
    file.close()


# 读取数据并删除非数为聚类做准备
df = pd.read_excel('./fooddata.xlsx')  # 读入表格数据
df1 = df.dropna()  # 删除含有数据缺失的行
# print(df1.head())  # 输出表格前5行`
data = df1.drop('食物名', axis=1, inplace=False)  # 删除'食物名'列 axis=0代表删除行,1代表删除列 inplace=False代表不改变原表 True代表改变原表
data = data.drop('序号', axis=1, inplace=False)  # 删除'序号'列
# print(data.head())


# 数据标准化
z_scaler = preprocessing.StandardScaler()
data_z = z_scaler.fit_transform(data)
data_z = pd.DataFrame(data_z)


# 数据归一化
minmax_scale = preprocessing.MinMaxScaler().fit(data_z)
dataa = minmax_scale.transform(data_z)
# print(pd.DataFrame(dataa).head())

idx,centroids = k_means(dataa, 8, 500)
label=[int(idx_item) for idx_item in idx]
# print(idx)
# print(centroids)
print(label)


data1 = df1['食物名']
data2 = data1.values


# 查看聚类结果
dat_type = pd.DataFrame(label)  # 将模型结果导出为数据表
dat_type.columns = ['类_别']  # 设置列名
dat = pd.merge(df1, dat_type, left_index=True, right_index=True)  # 合并类别表和数据表
# print(dat)
pd.set_option('display.max_rows', None)
dat.sort_values('类_别')  # 按类别进行排序
# print(dat.head(10))


# 储存聚类结果
FoodCluster = [[], [], [], [], [],[], [], []]  
for i in range(len(data2)):
    FoodCluster[label[i]].append(data2[i])

resultStr = ''  # 保存分类结果
# 输出分类结果
for i in range(len(FoodCluster)):
    print(FoodCluster[i])
    # 将同分类食物用,拼接
    resultStr = resultStr + ','.join(FoodCluster[i]) + '\n'
savaData('kmeans_resultF.csv', resultStr)
