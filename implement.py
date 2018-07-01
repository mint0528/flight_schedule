# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

from datetime import datetime as dt

from IPython.core.debugger import set_trace


# In[8]:


# read fs_0506.csv
parse_dates = ['date', 'de_time', 'la_time', 'flying_time']
fs_0506 = pd.read_csv('fs0506.csv', index_col='flight_id', parse_dates=parse_dates) # May 6th flight schedule


# In[9]:


fs_0506.flying_time = pd.to_timedelta(fs_0506.flying_time)


# In[10]:


# read initial_p1.csv
initial_p1 = pd.read_csv('initial_p1.csv', index_col=0) # initial pheromone matrix


# In[124]:
numant = 40 # 蚂蚁的个数

numcity = len(initial_p1.index) #城市的个数 582

alpha = 1  # 信息重要程度因子? 这是什么?

beta = 5 # 启发函数因子

rho = 0.1 # 信息素挥发程度

Q = 1 # 信息素常数

iter = 0

itermax = 200

# pheromone matrix
pheromonetable = initial_p1.copy()

# Pheromone index list
# make a index list(string) corresponds to Pheromone table's index(number)
name_list = pheromonetable.index.values
# 把Pheromone表index转化为数字保存成list

# change Pheromone_table into array in order to be consistant as original one.
pheromonetable = pheromonetable.values

pathtable = np.zeros((numant, numcity)).astype(int) # 路径记录表 (每一只蚂蚁走过的城市顺序) 40x582

lengthaver = np.zeros(itermax) # cost average per iteration 200x1

lengthbest = np.zeros(itermax) # 用来记录200次迭代的最佳长度, 也就是每次的最短长度. 200x1

pathbest = np.zeros((itermax, numcity)) # 每次迭代最短路径的城市顺序 200x582

P = [1, 2, 3, 4, 5] # This is five P values for p1, p2, p3, p4, p5.

num_plane = len(fs_0506.aircraft_id.unique()) # 121

cake = 1


def cancelled(translate_path):
    '''
    如果没有在tranlate_path中出现, 那么就是取消的航班.
    输入: list
    输出: list
    '''
    # translate_path 和 initial_p1 的 index 的 差.
    return list(set(initial_p1.index.values).difference(set(translate_path)))

def flight_air_match_dict(translate_path):
    '''
    Return two dicts refelct flight and airplane independently.
    Input: list ['axx', 'fxx', 'fxx', ... , 'axx'...]
    Output1: dict flight_airplane_dict {"fxx": "axx", ... "fxx": "axx"}
    Output2: dict airplane_flight_dict {"axx": ["fxx", ... , "fxx"], ... , "axx": ["fxx", ... , "fxx"]}
    '''
    transfered_list = ''.join(translate_path) # change list to one string.
    regex = r"(a\d+)((f\d+)+)"
    matches = re.finditer(regex, transfered_list)
    flight_airplane_dict = {} # {"fxx": "axx", ... "fxx": "axx"}
    airplane_flight_dict = {} # {"axx": ["fxx", ... , "fxx"], ... , "axx": ["fxx", ... , "fxx"]}
    for matchNum, match in enumerate(matches):
        a = match.group(1)
        f = match.group(2)
        regex = r"f\d+"
        f = re.findall(regex, f)
        flight_airplane_dict = merge_two_dicts(flight_airplane_dict, {i: a for i in f})
        airplane_flight_dict = merge_two_dicts(airplane_flight_dict, {a : f})
    return flight_airplane_dict, airplane_flight_dict

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def calculate_x(flight_airplane_dict, cancell_list):
    '''
    calculate x from flight_air_dict and cancell_list
    if cancelled , x = 0
    if new path is not consistent with old one, means airplane has been changed.
    input
    -----
    flight_airplane_dict: {"fxx": "axx", ... "fxx": "axx"}
    cancell_list: ["fxx", "fxx", ..., "fxx"]
    return
    ------
    x_dict: {"fxx": bool, ...}
    '''

    keys = flight_airplane_dict.keys()
    x_dict = {key: int(flight_airplane_dict[key] != fs_0506.loc[key].aircraft_id) for key in keys}
    x_dict = merge_two_dicts(x_dict, {cancelled_flight_id: 0 for cancelled_flight_id in cancell_list})
    return x_dict

# debug cancelled return []
# solutiuon: change position of diff.

def calculate_z(cancel_list):
    '''
    Funciton: get z dict from cancell list.
    Input
    -----
    cancel_list: list of flight ID ['fxx', ... , 'fxx']
    Return
    ------
    z_dict: dict ['flight': bool]
    '''
    z_dict = {key: 1 for key in cancel_list}

    z_dict = merge_two_dicts(z_dict, {key : 0 for key in list(set(fs_0506.index.values).difference(set(cancel_list)))})
    return z_dict

def calculatime_timestamp_interception(index1, index2, df=fs_0506, feature='la_time'):
    '''
    Function
    --------
    Given two filght id, calculate the latime delay between them.
    Input
    -----
    flight1, flight2: string 'fxxx'
    Output
    ------
    time: int (minutes)
    '''
    return (pd.to_datetime(df.loc[index1, feature]) - pd.to_datetime(df.loc[index2, feature])).seconds/60
# test
assert calculatime_timestamp_interception('f162', 'f10') == 705.0

def calculate_y(cancel_list, air_flight_dict, slack=50):
    '''
    Function: calculate y
    Input
    -----
    cancel_list: ['fxx', ... "fxx"]
    air_flight_dict: {'axx': ['fxx', ..., 'fxx']}
    delay_time: 50 min
    Return
    ------
    y_dict: {'flight': bool}
    delay_time: dict {'fxx': time}, time: int (minute)
    '''
    y_dict = {}
    delay_time = {}
    # I don't know how to write the rest of the code.
    # May be the little cake will tell me.
    # Or may not.
    # If you can't understand the code below, it's not my fault.
    # It's her fault.

    for item in air_flight_dict.items():
        a, f_list = item # a: 'axx', f_list: ['fxx', ... , 'fxx']
        y_dict = merge_two_dicts(y_dict, {f_list[0]: 0})
        delay_time = merge_two_dicts(delay_time, {f_list[0]: pd.to_timedelta(0, 'm')})
        for i in range(len(f_list)-1):
            fi = f_list[i]
            fj = f_list[i+1]
            y = y_dict[fi]
            if y == 0 :
                f1_de = fs_0506.loc[fi, 'de_time']
            elif y != 0:
                f1_de = fs_0506.loc[fi, 'de_time'] + delay_time[fi]
            f2_de = fs_0506.loc[fj, 'de_time']
            actual_la = f1_de + fs_0506.loc[fi, 'flying_time'] + pd.to_timedelta(slack, 'm')
            if  actual_la <= f2_de:
                y_dict[f_list[i+1]] = 0
                delay_time[f_list[i+1]] = pd.to_timedelta(0, 'm')
            else:
                y_dict[f_list[i+1]] = 1
                delay_time[f_list[i+1]] = actual_la - f2_de
    cancell_dict = {key:0 for key in cancel_list}
    delay_time = {k: int(value.seconds/60) for k, value in delay_time.items()}
    delay_time = merge_two_dicts(delay_time, cancell_dict)
    return merge_two_dicts(y_dict, cancell_dict), delay_time

# cost function
def calculate_xyz(path):
    """calculate x y z for cost funtion"""
    translate_path = [name_list[i] for i in path]
    cancelled_list = cancelled(translate_path)
    # print(cancelled_list)
    # print("cancell list length", len(cancelled_list))
    fad, afd = flight_air_match_dict(translate_path) # {flight:air} and {air:flight}
    x_dict = calculate_x(fad, cancelled_list)
    z_dict = calculate_z(cancelled_list)
    y_dict, delay_time_dict = calculate_y(cancelled_list, air_flight_dict=afd)
    assert len(delay_time_dict) == 461
    return x_dict, y_dict, z_dict, delay_time_dict

# def assign_func(translate_path):
#     '''
#     Input: list
#     step1: turn list of string into one string.
#     step2: find flight in time sequence.
#     Return: {'a19':['f2033', ...,'f1963']}
#     '''
#     # step1
#     transfered_list = ''.join(translate_path) # change list to one string.
#     x_dict = calculate_x(transfered_list)  # {flight_id: 1, ...., flight_id:0}
#     return x_dict
def cost(path, p=[15, 1e2, 1200, 1, 4]):
    """
    Input: path [0, 2, 3, ..., 68]
    Output: float
    """
    # cost is a combination of x, y, z, indicate flight status.
    p1, p2, p3, p4, p5 = p
    x, y, z, delay = calculate_xyz(path) # RETURN 4 dict. {'fxxx': bool}
    cost = 0
    for fid in fs_0506.index.values:
        cost = cost + x[fid] * p1 + delay[fid] *p2 + z[fid] * p3 + (y[fid] * delay[fid] *p4 +z[fid] *p5)* fs_0506.loc[fid, 'passenger_num'] + (y[fid] + z[fid]) * fs_0506.loc[fid, 'connect_passenger_num'] *p5
    return cost


# In[143]:


# test cost fucntion
_test_path = np.random.permutation(range(1, len(initial_p1.index[:-100])))
_test_path = np.insert(_test_path, 0, 0)
# print(_test_path)
# print("test path length is ", len(_test_path))
# test translate_path
calculate_xyz(_test_path)
# test assign_func step 1
# assign_func(calculate_xyz(_test_path))
# test cancelled function
# assert len(calculate_xyz(_test_path))==461
# test calculate z
# assert len(calculate_xyz(_test_path)) == 461
# test calculate y
# afd, y_dict, delay_time = calculate_xyz(_test_path)
# print(afd['a136'])
# for i in afd['a136']:
#     print("id:", i, "y:",y_dict[i], "delay time:", delay_time[i])
# fs_0506.loc[afd['a136']]
# len(y_dict)
# test cost
cost(_test_path)



# In[148]:


while iter < itermax:
    # 随机产生蚂蚁的起点城市
    if numant <= num_plane:
        pathtable[:, 0] = np.random.permutation(range(0, num_plane))[:numant] # 把起始城市按照蚂蚁数量随机分配
    else: # 对于蚂蚁比城市多的情况, 也是把所有城市随机分给每一个蚂蚁
        pathtable[:num_plane, 0] = np.random.permutation(range(0, num_plane))[:] # 先分配前面的蚂蚁, 保证每个城市都可能作为起点
        pathtable[num_plane:, 0] = np.random.permutation(range(0, num_plane))[:numant - num_plane] # 再分配多出的蚂蚁, 随机分配
    length = np.zeros(numant) # 蚂蚁的路径距离

    for i in range(numant): # 对于每一只蚂蚁
        # pathtable 40 x 52
        visiting = pathtable[i, 0] # 起始城市的位置, 就是之前随机分配的城市.
        visited = set() # 使用set记录所有去过的城市, 因为set是集合, 集合中的元素不可以重复.
        visited.add(visiting) # 每去一个城市, 把当前城市添加到集合里面.
        unvisited = set(range(numcity)) # 创建未访问的城市集合
        unvisited.remove(visiting) # 删除访问过的城市, 留下的就是没有访问的

        for j in range(1, numcity): # 访问完所有的城市需要的步数
            # 下面是轮盘法的代码, 使用轮盘法选择下一个城市
            # 轮盘法可能是一个选取城市的一个公式.
            listunvistited = list(unvisited)
            probtrans = np.zeros(len(listunvistited))

            for k in range(len(listunvistited)):
                # 这个下面就是那个很复杂的方程.t时刻,蚂蚁k从城市i到城市j的概率的分子
                # probtrans[k] = np.power(pheromonetable[visiting][listunvistited[k]], alpha)*np.power(etatable[visiting][listunvistited[k]], beta)

                probtrans[k] = np.power(pheromonetable[visiting][listunvistited[k]], alpha)
                # @TODO: 这里他写的是alpha, 按照公式应该是beta.
            # 计算概率
            if sum(probtrans) > 0:
                cumsumprobtrans = (probtrans/sum(probtrans)).cumsum()
            else:
                continue
            # 根据num_plane, 确保飞机全部飞出.
            # cumsumprobtrans[np.where(np.array(listunvistited)>=num_plane)] -= np.random.rand()
            # 概率随机减去一点
            cumsumprobtrans -= np.random.rand() # 减去的这一点可以让某些概率为负值.
            selected_city_index = np.where(cumsumprobtrans>0)[0][0]

            k = listunvistited[selected_city_index] # 下一个访问的城市. 城市为概率大于0的第一个城市.

            # @TODO: 这个地方可能会有bug, 评论里面说的
            pathtable[i,j] = k # 记录第i只蚂蚁在第j步访问的城市.
            unvisited.remove(k) # 移除访问过的城市
            visited.add(k)

            #length[i] += distmat[visiting][k] # 把蚂蚁之前城市到目前城市的距离算到总路程.
            visiting = k # 目前城市定位到k
            #length[i] = cost(path)
        #length[i] += distmat[visiting][pathtable[i, 0]] # 从目前的城市回归起点城市的距离.
        length[i] = cost(pathtable[i])
        # print("第{}蚂蚁总共走了{}米".format(i, length[i]))
    #set_trace()
    # 包含所有蚂蚁的一个迭代结束后, 统计这次迭代的统计参数.
    lengthaver[iter] = length.mean() # 所有的蚂蚁在这一个循环中走过的平均距离
    if iter == 0:
        lengthbest[iter] = length.min()
        pathbest[iter] = pathtable[length.argmin()].copy() # 走过路径最短的那个蚂蚁的路线作为最佳路线
    else:
        if length.min() > lengthbest[iter-1]: # 如果当前的最短路径大于上一次迭代的最短路径
            lengthbest[iter] = lengthbest[iter-1] # 最短的路径依然是上次迭代的结果
            pathbest[iter] = pathbest[iter-1].copy() # 最佳路径依然是之前的路径
        else: # 如果当前路径最佳,更新最佳路径和最佳距离
            lengthbest[iter] = length.min()
            pathbest[iter] = pathtable[length.argmin()].copy()
            print("最短路径{}米".format(lengthbest[iter]))
            print("经过城市{}".format(pathbest[iter]))
    # 更新信息素
    change_pheromone_table = np.zeros((numcity, numcity))
#     for i in range(numant):
#         for j in range(numcity-1):
#             change_pheromone_table[pathtable[i,j]][pathtable[i, j+1]] += Q/distmat[pathtable[i,j]][pathtable[i, j+1]]
#         change_pheromone_table[pathtable[i, j+1]][pathtable[i, 0]] += Q/distmat[pathtable[i, j+1]][pathtable[i, 0]] # 从最后一个城市回到起点
#     pheromonetable = (1-rho)*pheromonetable + change_pheromone_table

    for i in range(numant):
        for j in range(numcity-1):
            change_pheromone_table[pathtable[i,j]][pathtable[i, j+1]] += Q/length[i]*cake
    pheromonetable = (1-rho)*pheromonetable + change_pheromone_table
    iter += 1

    if (iter-1)%20 == 0:
        print("当前迭代次数{}".format(iter-1))
