# NodeClassification

> 
**모델 목표 : 이미지 간 유사 여부 판별 모델**
  
> 문제를 단순 graph classfication 으로 축소하여 진행함  
> 모델의 아웃풋으로 15개의 클래스 중에서 가장 유사한 클러스터링 값을 예측하면, 원래 라벨과 비교해서 학습하고,  
범위1과 범위 2의 예측 클래스 번호 간 비교를 통해 TF를 return 함   

> 해당 예제에서는 df를 이용해 아래와 같이 결과를 확인함
> <img src = "https://github.com/Hanin00/Image_Retrieval/blob/9d3a21300c012d31c1e35474f19a648f14d14fbd/extraImages/NodeClassification-CompareResult.png">


> 

## Model 1
- sub와 obj  간에 relationship이 반영이 되지 않은 모델  
- Adjacency Matrix의 선정에 오류 있음
>   


### **Input data format**
> **Adj** : id x id (같은 cluster 값을 갖는 경우 1로 체크해 인접 관계를 나타냄) (1000, 1000)  
>**Feature** : id x freObj(해당 img의 relationship을 갖는 Obj 또는 subject내에  freObj가 있는 경우 1, 없는 경우 0) (1000, 100)  
>   - freObj : 대상이 되는 이미지 1000개의 Scene graph에서 가장 언급량이 많은 Obj 100개      
>   
> **Label** : 각 id 당 cluster 번호, 총 15개의 cluster 종류가 있음 (1000, 1)    
>  - cluster는 bert-base-nli-mean-tokens를 이용해 15개의 클러스터로 분류했음   
>  - region_graph.json에서 이미지 1000개의 phrase 값을 embedding 해 유사 여부를 판단하고 clustering 과정을 통해 15가지 종류로 이미지를 분류함    
      이를 통해 Label 값을 생성함  
>- realtionship이 잘 나타날 수 있도록 개선 필요
   
  
  > 



## Model 2(Node Classification) : 
- Adj Matrix가 인접을 나타내지 않음   

### **Input data format**
> **Adj** : ImageId x FreObj (ImageId에 따른 object의 언급을 나타냄) (1000, 100)  
>**Feature** : freObj의 Fasttext Embedding (100,10)
>  - freObj Column내에서 freObj의 각 단어를 FastEmbedding을 통해 특징 지음 
> 
>**Label** : 각 id 당 cluster 번호, 총 15개의 cluster 종류가 있음 (1000, 1)   
>  - model 1과 동일함
  
  >

----  
> 
## Code 관련 내용
> **model 1**
> - main.py  : GCN과 Util을 이용, 모델 구현 및 epoch 설정 등에 관한 내용
> - util.py :  load data(), train(), evaluate() 및 test() 등 관련 함수
> - GCN.py : 모델 정의
>
> **model 2**
> - NodeClass2.py : Model 1의 GCN.py와 util.py를 동일하게 사용하며, Input data 변경 관련한 수정
 
>


## **Failure Log**

- 모델 1의 경우 id의 cluster값을 안다는 전제가 필요함 *-> Node Classification이라고 할 수 없음  
- Model 2의 경우 accuracy가 매우 낮은 것을 알 수 있음


**개선에 관한 의견**
> - 아래 그림은 id 값에 따른 freObj의 분포를 Seaborn의 Heatmap을 이용해 구현한 도표임   같이 데이터의 특징이 고루 분포하지 않고 뚜력하게 분류되지 않음을 알 수 있음  데이터 전처리 개선 또는 특징을 더 반영할 수 있는 Feature Map이 필요함
> <img src= "https://github.com/Hanin00/Image_Retrieval/blob/598abc6b1d7eb0ab8d777ee686156dcd514051a8/extraImages/dataHeatMap.png">




>

응용 코드 : https://github.com/fahim-sikder/Node-Classification-GCN

