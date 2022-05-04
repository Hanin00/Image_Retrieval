import numpy as np
import pandas as pd
import json
from openpyxl import Workbook
from gensim.models import FastText
from tqdm import tqdm
from collections import Counter
import torch

''' 데이터 전처리 관련 utility '''



np.set_printoptions(linewidth=np.inf)

''' 1000개의 이미지의 빈출 objName '''
def adjColumn(imgCount):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(imgCount):
            objects = data[i]["objects"]
            for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
                object.append(objects[j]['names'])
        object = sum(object, [])
        count_items = Counter(object)
        frqHundred= count_items.most_common(n=1000)
        adjColumn = []
        for i in range(len(frqHundred)):
            adjColumn.append(frqHundred[i][0])
        return adjColumn


''' adj 생성(이미지 하나에 대한) 
    100x100으로 변경해야함
'''



def createAdj(imageId, adjColumn, sceneGraph, objJson, ):
    adjM = np.zeros((len(adjColumn), len(adjColumn)))
    # 이미지 내 object, subject 끼리 list 만듦. 한 relationship에 objId, subId 하나씩 있음. Name은 X


    # imgId의 relationship에 따른 objId, subjId list
    # i는 image id
    # imageDescriptions = data[imageId-1]["relationships"]
    imageDescriptions = sceneGraph[imageId - 1]["relationships"]
    objectId = []
    subjectId = []

    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        objectId.append(imageDescriptions[j]['object_id'])
        subjectId.append(imageDescriptions[j]['subject_id'])
    # object = sum(object, [])

    # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출
    # obj에서 각 id로 objName, subName 찾아서 리스트로 저장
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
    # imgId의 relationship에 따른 objId, subjId list
    # i는 image id
    # objectId = data[imgId][""]

    # 한 이미지 내에서 사용되는 obj의 Id 와 이름 dict;  여러 관계 간 동일 obj가 사용되는 경우가 있기 때문
    # subject의 id값을 넣었을 때 name이 제대로 나오는 지 확인 :
    objects = objJson[imageId - 1]["objects"]
    allObjName = []
    for i in range(len(objects)):
        allObjName.append(([objects[i]['names'][0]], objects[i]['object_id']))
        if not objects[i]['merged_object_ids'] != []:  # id 5090처럼 merged_object_id에 대해서도 추가해주면 좋을 듯
            for i in range(len(objects[i]['merged_object_ids'])):
                allObjName.append(([objects[i]['merged_object_ids'][0]], objects[i]['object_id']))

    objIdName = []
    subIdName = []
    for i in range(len(subjectId)):
        objectName = ''
        subjectName = ''
        for mTuple in allObjName:
            if objectId[i] in mTuple:
                objectName = str(mTuple[0][0])
            if subjectId[i] in mTuple:
                subjectName = str(mTuple[0][0])
            if (objectName != '') & (subjectName != ''):
                objIdName.append(objectName)
                subIdName.append(subjectName)
    # 위에서 얻은 obj,subName List로 adjColumn인 freObj에서 위치를 찾음
    for i in range(len(objIdName)):
        adjObj = ''
        adjSub = ''
        if objIdName[i] in adjColumn:
            adjObj = adjColumn.index(objIdName[i])
            adjM[adjObj][adjObj] += 1
        if subIdName[i] in adjColumn:
            adjSub = adjColumn.index(subIdName[i])
            adjM[adjSub][adjSub] += 1
        if (adjObj != '') & (adjSub != ''):
            adjM[adjObj][adjSub] += 1
    adjM = torch.Tensor(adjM)

    return adjM


''' obj name 단순 임베딩(fasttext로 임베딩 한 값)'''
def objNameEmbedding(xWords):
    a = []
    a.append(xWords)
    # model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText(xWords, vector_size=10, workers=4, sg=1, word_ngrams=1)

    # for i in a :
    embedding = []
    for i in xWords:
        embedding.append(model.wv[i])
    return embedding


''' 
feature matrix 2안 
scene graph에서 object-predicate-subject를 scenetence로 묶어서 임베딩 
-> 질문 : 이때 각 단어에 대한 임베딩은 어케 구할건지? 
    일일이 비교해서 구해야 하는지? 
    word가 아니고 phrase인 경우에는? 
    padding?
    '''

''' 0503 Dataset 추가'''
# freObj 생성 시 대상 이미지 수 1000-> 5000/10000 변경
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []
    for i in range(10000):
        objects = data[i]["objects"]
        for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
            object.append(objects[j]['names'])
    object = sum(object, [])
    count_items = Counter(object)
    frqHundred = count_items.most_common(500)
    adjColumn = []
    for i in range(len(frqHundred)):
        adjColumn.append(frqHundred[i][0])

    with open('./data/freObj5000_500.txt', 'w') as file:
       file.writelines(','.join(adjColumn))

testFile = open('./data/freObj5000_500.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
list = (readFile[1:-1].replace("'",'')).split(',')

print(len(list))
print(list[0])


# '''freObj Embedding txt 저장'''
# testFile = open('./data/freObj5000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# freObj = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
# freObj = freObj[:100]  # 빈출 100 단어 만 사용

# freObjEmbedding = objNameEmbedding(freObj)
# freObjEmbedding = torch.tensor(freObjEmbedding)

# with open('./data/freEmb5000.txt', 'w') as file:
#        file.writelines(','.join(adjColumn))


# f1 = open('./data/imgAdj5000.txt', 'w')
# for i in range(5000):
#     df_adj, adjMatrix = createAdj(i, adjColumn)
#     f1.write(adjMatrix+",")
# f1.close()