import numpy as np
import pandas as pd
import json

import torch
from openpyxl import Workbook
import util as ut
from gensim.models import FastText
from tqdm import tqdm
import util as ut
import YEmbedding as yed

np.set_printoptions(linewidth=np.inf)

''' 1000개의 이미지에 존재하는 obj_id(중복 X) '''
def adjColumn(imgCount):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(imgCount):
            imageDescriptions = data[i]["objects"]
            for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
                object.append(imageDescriptions[j]['object_id'])
        scene_obj_id = sorted(list(set(object)))

        return scene_obj_id


''' adj 생성(이미지 하나에 대한) '''
def createAdj(imageId, adjColumn):
    adjM = np.zeros((len(adjColumn), len(adjColumn)))
# 이미지 내 object, subject 끼리 list 만듦. 한 relationship에 objId, subId 하나씩 있음. Name은 X

    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        # imgId의 relationship에 따른 objId, subjId list
        # i는 image id
        # imageDescriptions = data[imageId-1]["relationships"]
        imageDescriptions = data[imageId-1]["relationships"]
        objectId = []
        subjectId = []

        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            objectId.append(imageDescriptions[j]['object_id'])
            subjectId.append(imageDescriptions[j]['subject_id'])
        # object = sum(object, [])

        # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출
    # obj에서 각 id로 objName, subName 찾아서 리스트로 저장
    with open('./data/objects.json') as file:  # open json file
        data = json.load(file)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
        # imgId의 relationship에 따른 objId, subjId list
        # i는 image id
        # objectId = data[imgId][""]

        # 한 이미지 내에서 사용되는 obj의 Id 와 이름 dict;  여러 관계 간 동일 obj가 사용되는 경우가 있기 때문
        # subject의 id값을 넣었을 때 name이 제대로 나오는 지 확인 :
        objects = data[imageId-1]["objects"]
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

   # adjM = torch.Tensor(adjM)
    return adjM


''' 
Y data 생성을 위해 image에 대한 text description을 이미지 별로 모음

jsonpath : './data/region_descriptions.json'
xlxspath : './data/image_regions.xlsx'
'''
def jsontoxml(imgCnt, jsonpath, xlsxpath) :
    with open(jsonpath) as file:  # open json file
        data = json.load(file)
        wb = Workbook()  # create xlsx file
        ws = wb.active  # create xlsx sheet
        ws.append(['image_id', 'region_sentences'])
        phrase = []

        q = 0
        for i in data:
            if q == imgCnt:
                break
            regions = i.get('regions')
            imgId = regions[0].get('image_id')
            k = 0
            for j in regions:
                if k == 7:
                    break
                phrase.append(j.get('phrase'))
                k += 1
            sentences = ','.join(phrase)
            ws.append([imgId, sentences])
            phrase = []
            q += 1
        wb.save(xlsxpath)


''' obj name 단순 임베딩(fasttext로 임베딩 한 값)'''
def objNameEmbedding(xWords) :
    a = []
    a.append(xWords)
  #  model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText(xWords, vector_size=10, workers=4, sg=1, word_ngrams=1)

    # for i in a :
    embedding = []
    for i in xWords:
        embedding.append(model.wv[i])

    return embedding


''' 1000개의 이미지에 존재하는 obj_name(중복 X) > Featuremap object_name Embedding 원본
    object_id, object_name의 개수가 일치하지 않는 문제 -> 동일 id에 이름 두 개씩 들어가 있는 경우 발견
    -> 이름을 합침 '''

def adjColumn_kv(imgCount):
    with open('../data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        dict = {}
        # dict 생성(obj_id : obj_name)
        for i in range(imgCount):
            imageDescriptions = data[i]["objects"]
            for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
                obj_id = imageDescriptions[j]['object_id']
                if (len(imageDescriptions[j]['names']) != 1):
                    wholeName = str()
                    for i in range(len(imageDescriptions[j]['names'])):
                        wholeName += imageDescriptions[j]['names'][i]
                    lista = []
                    lista.append(wholeName.replace(' ', ''))
                    obj_name = lista
                else:
                    obj_name = imageDescriptions[j]['names']

                dict[obj_id] = obj_name
            #  print(obj_id)
            #  print(obj_name)

        keys = sorted(dict)
        val = []

        for i in keys:
            if (type(dict[i]) == str):
                val += (dict[i])

            val += dict[i]
        return keys, val




def createAdj_model2(imageId, adjColumn, sceneGraph, objJson, ):
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



''' 
feature matrix 2안 
scene graph에서 object-predicate-subject를 scenetence로 묶어서 임베딩 
-> 질문 : 이때 각 단어에 대한 임베딩은 어케 구할건지? 
    일일이 비교해서 구해야 하는지? 
    word가 아니고 phrase인 경우에는? 
    padding?
    '''


