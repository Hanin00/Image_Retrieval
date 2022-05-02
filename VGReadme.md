# **Visual Genome**

https://visualgenome.org/
> 
> 이미지 상의 범위를 특정해 캡션을 달아 object-predicate-subject의 relationship을 가진 Scenegraph를 제공함
> - Total Image : 108,077  
> - Total region descriptions: 4,297,502  
> - Total image object instances: 1,366,673  
> - Unique image objects: 75,729  
> ⇒ ObjectId는 개별적이며 Object Name은 동일함

 
## **Used Data**
> - Node Classification / Graph Classification 은 Image 1000개를 대상으로 함  
> - 각 json 형식은 https://visualgenome.org/api/v0/api_readme 참고


### region_graph.json
- 각 이미지의 범위에 대한 설명(Phrase)를 사용해 Image를 Cluster로 분류할 때 사용함 -> [util readme  참고](https://github.com/Hanin00/Image_Retrieval/blob/73f0217105e55576bde127848e787a22f0716dc0/UtilsReadme.md)
- Object는 Object name이 아닌 id로만 제공됨  

<img src="https://github.com/Hanin00/Image_Retrieval/blob/82326deea0e8b8e4092b23ca5b8116548a6f8054/extraImages/region_graphJsonStructure.PNG">

### scene_graph.json
- 이미지 내 Object 간의 연결 관계를 Relationships를 통해 나타냄
- region_graph.json과 달리 phrase나 bounding box의 좌표가 없고, objects.json과 달리 object Name이 없음

<img src="https://github.com/Hanin00/Image_Retrieval/blob/82326deea0e8b8e4092b23ca5b8116548a6f8054/extraImages/scene_graphsJsonStructure.PNG">

### objects.json
- 동일 이미지 Id에 속한 Object의 Id와 object를 특정하는 bounding box의 왼쪽 상단의 x,y 좌표 및 w, h와 Name, synset을 제공함
- 이미지 내의 관계를 이용하기 위해 사용한 region_graph와 scene_graph에는 object Name이 없어 Object.json내의 objectId를 비교해 Object name을 얻는 방법으로 사용함

<img src="https://github.com/Hanin00/Image_Retrieval/blob/82326deea0e8b8e4092b23ca5b8116548a6f8054/extraImages/objectStructure.PNG">

이미지 출처 : visualGenome https://visualgenome.org/api/v0/api_readme
 
