import json
from openpyxl import Workbook
import util as ut

'''         subjectName.append(xWords[adjColumn.index(subjectId[j])])
            objectName.append(xWords[adjColumn.index(objectId[j])])
'''
imgCnt = 1000
jsonpath = './data/scene_graphs.json'
xlsxpath = './data/scene_sentence.xlsx'

adjColumn, xWords = ut.adjColumn_kv(imgCnt)

with open(jsonpath) as file:  # open json file
    data = json.load(file)
    wb = Workbook()  # create xlsx file
    ws = wb.active  # create xlsx sheet
    ws.append(['image_id', 'scene_sentence', 'used_ids'])
    objName = []
    objId = []
    predicate = []
    predicateId = []

    # relationship에서 relationshipId, predicate, subId, objId 구함-> subId, objId를 adjColumn에서 idx 찾고 그 idx로 xWords 찾아서 append
    # realtionship Id, subId, objId  합쳐서 used_ids에 append / predicate, subId, objId 합쳐서 scene_sentence에 append
    # 이미지 id 하나 넘어가면 scene_sentence, used_ids = []

    q = 0
    sentences = usedId = []
    for i in data:
        if q == imgCnt:
            break
        # image_id
        imgId = i.get('image_id')
        #relationship
        relationships = i.get('relationships')
        objects = i.get('objects')
        objDict = {}

        maxlen = 0
        #objId, Name 목록
        for k in range(len(objects)) :
            objDict[objects[k].get('object_id')] = objects[k].get('names')

        predicateId=subjectId= objectId = []
        predicate= subjectName= objectName = []
        for j in range(len(relationships)) :
            predicateId.append(relationships[j].get('relationship_id'))
            predicate.append(relationships[j].get('predicate'))
            subjectId.append(relationships[j].get('subject_id'))
            objectId.append(relationships[j].get('object_id'))

            relationships[j].get('object_id')
            objects(relationships[j].get('object_id'))
            sentences.append(predicate + subjectName + objectName)
            usedId.append(predicateId + subjectId + objId)

        ws.append([imgId, sentences, usedId])
        sentences, usedId = []
        q += 1
    wb.save(xlsxpath)


'''
    q = 0
    for i in data:
        if q == imgCnt:
            break
        #image_id
        imgId = i.get('image_id')
        #objName, subjectName
        object = i.get('objects')
        objName = []
        objId = []
        predicate = []
        predicateId=[]
        for j in object :
                objName.append(object[j].get('name'))
                objId.append(object[j].get('object_id'))
        print("len(objName) : ", len(objName))
        print("len(object_id) : ", len(objId))
        # preicate
        k = 0
        relationships = i.get('relationships')
        for j in relationships:
            if k == 2:
                break
            predicate.append(j.get('predicate'))
            predicateId.append(j.get('relationship_id'))
            k += 1
        sentences = ','.join(phrase)
        ids = objId+predicateId
        ws.append([imgId, sentences,])
        phrase = []
        q += 1
    wb.save(xlsxpath)
'''