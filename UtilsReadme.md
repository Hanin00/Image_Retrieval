# **Util, Util2, YEmbedding**


**util, util2 파일은 프로젝트 모두 동일하나 graphClassfication 폴더 내의 파일을 기준으로 함**

> Util.py : Related with GCN Training & validation   
> Util2.py : Related with data preprocessing   


## Util.py
> Related with GCN Training & validation
>

#### normalize(mx)


#### splitDataset(seed, n_train, n_val, n_features) 
- random seed를 지정해 예제 확인 시 비슷한 결과가 나오도록 함


#### toDevice


#### train

#### evaluate

#### accuracy

#### loadData







## Util2.py
> Related with data preprocessing

#### adjColumn(imgCount)
- imgCount 만큼의 이미지에서 자주 언급되는 단어들을 최빈순으로 정렬해 list를 반환함  
- scene_graph의 object Id를 이용했고 Object.json에서 object Name을 가져옴
  
> 

#### create_adjMatrix(clusterList) 
- 대상이 되는 이미지 개수 1000 x 1000의 영행렬을 만들고, 각 이미지 간 클러스터 값이 같은 경우 1을 추가해 이미지 간의 연관성을 표현함. 이에 대각행렬을 더함
- node classification model 1에 사용
  
> 

#### createAdj(imageId, adjColumn, sceneGraph, objJson) 
- adjColum : freObj List
- 이미지 하나(imageId)에 대한 Adj matrix 생성 (100 x 100)
1. 한 이미지 내의 relationship에 따른 objectId, subjectId List를 만들고(sceneGraph 사용), Id에 따른 Name List를 생성함(Obj Json 사용)
2. (len(adjColumn), len(adjColumn))크기의 0 행렬을 생성함
3. subJectName, objectName의 각 원소가 adjColumn과 일치하는 경우 += 1로 특징을 추출함
4. GCN의 Feauture Matrix Input data는 tensor type이므로 형변환 해 return 함
- (len(adjColumn), len(adjColumn)) 으로 해당 모델에서는 100 x 100으로 함
- graph classification model 1에 사용

  
> 

#### featuremap(startId, endId, freObjList) 
- 이미지 하나 안의 obj랑 freObjList와 일치하면 +=1로 표시해서 특징을 나타냄
- 1000,100
- node classification model 1의 featuremap, node classification model 2의 adjMatrix로 사용함

> 

#### objNameEmbedding(freObjList)
- Fasttext를 이용해 List 내의 각 object Name을 임베딩함

>  


## YEmbedding
1000개의 이미지 별로 region_graph 의 phrase 값을 embedding 하고, 각 이미지를 15개의 클러스터로 분류함
하단 링크의 bert-base-nli-mean-tokens 사용
ref) https://github.com/marialymperaiou/visual-genome-embeddings

>  
-------
### 모델 별 사용 함수
Node Classification - Adj, Feature
- model 1 : create_adjMatrix(), featuremap()
- model 2 : featuremap(), objNameEmbedding()

Graph Classification 
- model 1 : createAdj(), featuremap()
- model 2 : 


