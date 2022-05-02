# Image_Retrieval
> scene_graph를 사용하여 GNN을 통해 유사 이미지 검색
---
## 1. Experiment Plan
#### torch를 이용한 GNN 구현
- 학습 데이터 전처리
  - Visual Genome을 이용하여 학습 데이터 생성
- GNN 모델 확인 및 테스트(~4.11)
  - Cora dataset을 이용하여 기존 GNN 모델 공부
  - 기존 GNN 모델의 사용하는 input data 형태 확인
- 학습 및 검증
  - 학습 데이터와 기존 GNN 모델인 GCN을 이용하여 유사 이미지 분류
    - NodeClassification (~4.13)
    - GraphClassficiation (~4.15)
    
  - Accuracy 및 Loss 확인 및 검증
---

## 2. Issue
#### ~ 22.04.05)
> - 가장 많이 사용되는 object id를 이용하여 인접행렬 (100 x 100) 생성
> - object id들의 대해 Embedding을 통해 feature 행렬(100 x 10) 생성
> - image들에 대해 k-mean 클러스터링을 수행하여 y값 생성

#### 22.04.06)
> - GCN 모델에 input의 포맷이 어떻게 되어있는지 파악 부족
> - CCN 모델에서 Graph Convolution을 어떻게 코드로 구현되었는지의 대한 파악 부족
> - 4.5/화 까지 마무리 하기로 했으나 기존 GCN 모델 사용에서 문제가 생겨 지연   

#### 22.04.11)
> - Cora Dataset을 이용한 GCN 예제 코드 분석
>   - Input, Output type 및 shape
>   - 코드의 기능   

#### 22.04.12)
> - Cora Dataset을 이용한 CoraGCN code Upload
> - Cora Dataset을 이용한 CoraGCN code 주석 정리
> - Visual Genome Dataset 분석 내용 정리
> - Visual Genome Dataset을 이용한 Node Classification Readme 정리
> - Visual Genome Dataset 전처리 관련 Utility Readme 정리
> - CoraGCN code를 이용한 Node Classification 학습 및 검증 -> data quality issue(epoch 1000 - acc:0.160)



>__개선사항__  
> - [ ] CORA dataset을 이용한 GNN 방법 정리

---

