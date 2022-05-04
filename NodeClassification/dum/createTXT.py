import sys
import csv
import time

import util as ut
#import YEmbedding as yed
import numpy as np
import pandas as pd
import json
import sys
import torch

# startId = 1
# endId = 1000
# #xlxspath = './data/image_regions.xlsx'
#
# testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# label = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
# label = label[:1000]  # 클러스터 1000개


# # Y - image, cluser 몇 번인지~
# embedding_clustering = yed.YEmbedding(xlxspath)
# idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# label = idCluster['cluster']
# df = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# print(df)
#
# print(label)



# print(label)
# freObj = ut.prequency_feature(1, 1000)
# adjMatrix = ut.create_adjMatrix(clusterList=label[1])
# # featuremap =ut.featuremap(startId,endId,freObj)


# listitem = label[1]
# output_array = np.array(listitem)

'''label txt 저장'''
# xlxspath = './data/image_regions.xlsx'
# # Y - image, cluser 몇 번인지~
# embedding_clustering = yed.YEmbedding(xlxspath)
# idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# label = idCluster['cluster']
# j = label.tolist()
# print(type(j))
# print(j)
# list_a = list(map(str, j))
#
'''txt 로 저장'''
# with open('cluster.txt', 'w') as file:
#    file.writelines(','.join(list_a))
#
# 


''' 1000개의 이미지의 최빈 objName 100개'''
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#     object = []
#     for i in range(1000):
#         objects = data[i]["objects"]
#         for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
#             object.append(objects[j]['names'])
#     object = sum(object, [])
#     count_items = Counter(object)
#     frqHundred = count_items.most_common(100)
#     adjColumn = []
#     for i in range(len(frqHundred)):
#         adjColumn.append(frqHundred[i][0])
# 
#     with open('freObj.txt', 'w') as file:
#        file.writelines(','.join(adjColumn))
# 
# testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'",'')).split(',')
# 
# print(len(list))
# print(list[0])


'''txt 불러오기'''
# testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'",'')).split(',')


'''id x id 동일 클러스터 Adj 저장'''
# adjM = np.zeros((len(imgCnt), len(imgCnt))) #1000x1000(id 개수로 해야함. 근데 테스트라 10개만)
# adjM = np.zeros((1000, 1000)) #1000x1000(id 개수로 해야함. 근데 테스트라 10개만)
# for i in range(len(adjM[0])):
#     for j in range(len(adjM[0])):
#         if cLabel[i] == cLabel[j]:
#             adjM[i][j] += 1
#         if i == j:
#             adjM[i][j] += 1
# np.save('idAdj.npy',adjM)
# idAdj = np.load('idAdj.npy')
#


#
# '''img 당 freObj 있/없 featuremap-image 1'''
# testFile = open('./data/freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'",'')).replace(' ','').split(',')
#
# freObj = list[:100]
# adjM = np.zeros((1000, 1000))
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#     object = []
#     for i in range(1000):  # 이미지 1000개에 대한 각각의 objectNamesList 생성
#         objects = data[i]["objects"]
#         for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
#             object.append(objects[j]['names'])  # 이미지 하나에 대한 objList
#
#        # object = sum(object, [])
#
#         # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출
#
#         row = [0 for i in range(len(freObj))]
#         l = 0
#         for k in range(len(freObj)) :
#             for j in range(len(objects)):
#                 n = ''.join(object[j])
#                 m = freObj[k]
#
#                 if n in freObj :
#                     w = freObj.index(n)
#                     row[w] = 1
#
#         featureMatrix.append((row))
#
#
# '''img 당 freObj 있/없 featuremap 이미지 하나 당'''
# testFile = open('./data/freObj.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'", '')).replace(' ', '').split(',')
#
# freObj = list[:100]
# adjColumn = freObj
# adjM = np.zeros((len(adjColumn), len(adjColumn)))
# # 이미지 내 object, subject 끼리 list 만듦. 한 relationship에 objId, subId 하나씩 있음. Name은 X
# imageId = 1
#
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#     # imgId의 relationship에 따른 objId, subjId list
#     # i는 image id
#     # imageDescriptions = data[imageId-1]["relationships"]
#     imageDescriptions = data[0]["relationships"]
#     objectId = []
#     subjectId = []
#
#     for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
#         objectId.append(imageDescriptions[j]['object_id'])
#         subjectId.append(imageDescriptions[j]['subject_id'])
#     # object = sum(object, [])
#
#     # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출
# # obj에서 각 id로 objName, subName 찾아서 리스트로 저장
# with open('./data/objects.json') as file:  # open json file
#     data = json.load(file)
#     # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
#     # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
#     # imgId의 relationship에 따른 objId, subjId list
#     # i는 image id
#     # objectId = data[imgId][""]
#
#     # 한 이미지 내에서 사용되는 obj의 Id 와 이름 dict;  여러 관계 간 동일 obj가 사용되는 경우가 있기 때문
#     # subject의 id값을 넣었을 때 name이 제대로 나오는 지 확인 :
#     objects = data[0]["objects"]
#     allObjName = []
#     for i in range(len(objects)):
#         allObjName.append(([objects[i]['names'][0]], objects[i]['object_id']))
#         if not objects[i]['merged_object_ids'] != []:  # id 5090처럼 merged_object_id에 대해서도 추가해주면 좋을 듯
#             for i in range(len(objects[i]['merged_object_ids'])):
#                 print(objects[i]['merged_object_ids'][i])
#                 allObjName.append(([objects[i]['merged_object_ids'][0]], objects[i]['object_id']))
#
#     print(allObjName)
#     print(type(allObjName[0][0]))
#
# objIdName = []
# subIdName = []
# for i in range(len(subjectId)):
#     objectName = ''
#     subjectName = ''
#     for mTuple in allObjName:
#         if objectId[i] in mTuple:
#             objectName = str(mTuple[0][0])
#         if subjectId[i] in mTuple :
#             subjectName = str(mTuple[0][0])
#         if (objectName != '') & (subjectName != '') :
#             objIdName.append(objectName)
#             subIdName.append(subjectName)
#
# '''if objectName in allObjNameDict :
#     objectName = allObjNameDict[objectId[i]]
# if subjectName in allObjNameDict :
#     subjectName = allObjNameDict[subjectId[i]]
# if (objectName!='')&(subjectName!=''):
#     objIdName.append(objectName)
#     subIdName.append(subjectName)'''
#
# # 위에서 얻은 obj,subName List로 adjColumn인 freObj에서 위치를 찾음
# for i in range(len(objIdName)):
#     adjObj = ''
#     adjSub = ''
#     if objIdName[i] in adjColumn:
#         print(objIdName[i])
#         adjObj = adjColumn.index(objIdName[i])
#         adjM[adjObj][adjObj] += 1
#     if subIdName[i] in adjColumn:
#         adjSub = adjColumn.index(subIdName[i])
#
#         adjM[adjSub][adjSub] += 1
#         print(subIdName[i])
#     if (adjObj != '') & (adjSub != ''):
#         adjM[adjObj][adjSub] += 1
#
# #np.set_printoptions(sys.maxsize)
# print(adjM)
#
#

# '''freObj Embedding txt 저장'''
# testFile = open('./data/freObj.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# freObj = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
# freObj = freObj[:100]  # 빈출 100 단어 만 사용
#
# freObjEmbedding = ut.objNameEmbedding(freObj)
# freObjEmbedding = torch.tensor(freObjEmbedding)






# ''' image 별로 freObjxfreObj 만들어서 dataset 만듦'''
# ''' id, Adj, label List 만드는 코드 '''
# # 빈출 단어 값
# testFile = open('../data/freObj.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# freObj = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
# freObj = freObj[:100]  # 빈출 100 단어 만 사용
#
# testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# label = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
# label = label[:1000]  # 클러스터 1000개
#
# # # 임베딩값 freObj x embedding(10)
# # feature = ut.objNameEmbedding(label)
#


testFile = open('../data/freObj5000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObj = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
freObj = freObj[:100]  # 빈출 100 단어 만 사용

with open('../data/scene_graphs.json') as file:
    data1 = json.load(file)
with open('../data/objects.json') as file:  # open json file
    data2 = json.load(file)

import pickle
dataset = []
start = time.time()
for i in range(5000):
    adj = ut.createAdj_model2(i, freObj,data1, data2)
    #dataset.append((i+1, adj, label[i]))
    dataset.append((adj))

print("obj : ", time.time() - start)

## Save pickle
with open("../data/feature_model2.pickle", "wb") as fw:
    pickle.dump(dataset, fw)

## Load pickle
with open("../data/feature_model2.pickle", "rb") as fr:
    data = pickle.load(fr)
print(data)


    # dataset.appen([i + 1, adj, label[i]] )e

#np.save('./data/frexfre1000.npy', np.ndarray(dataset))
#np.save('./data/frexfre1000.npy', np.ndarray(adj))
