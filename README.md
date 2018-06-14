# RecSys Project: Structured SLIM
### Intro:

The aim of this project is to learn the weights to assign at each relevant attribute of a given dataset in order to maximize the MAP performance, exploiting SLIM Elasticnet algorithm.

In order to obtain the relative attributes' weight, the algorithm train a model that minimizes the difference between the similarity matrix obtained by a CF method and the ones obtained by a CBF approach. During this process, the only cells that are updated, and from which the algorithm learns, are the ones related to items with common features.

Given a similarity matrix S (random initialized) and a URM matrix, SR = R for each element filtered by the mask. The Learning approach is the classical SGD but is applied only to the relevant items. The error is computed as follow:

![equation](http://latex.codecogs.com/svg.latex?e=\sum_{k}(S_{ik}\cdot{R_{-ik}})-r)

and consequently:

![equation](http://latex.codecogs.com/svg.latex?s_{ik}=\lambda\cdot{e}+\gamma+2\beta{S_{ik}})

### Parameters setting:

As we said, in order to create the structure that selects the cells to update, a similarity matrix is computed starting from the ICM. For this reason is also possible to set the parameters that regulate this operation: topK and Shrinkage.

The algorithm takes as input two matrixes: ICM and URM in *scipy.sparse.csc_matrix* format.

In the fitting phase is possible to set: 

top K, shrinkage and a boolean for the normalization used for the structure creation, learning rate λ, l2-norm coefficient β, l1-norm coefficient γ, and epochs number used for the learning process.

### Basic usage:

```python
import SLIM_Elastic_Net_Structured

URM_train = sps.load_npz("files/train.npz")
URM_test = sps.load_npz("files/test.npz")
ICM = sps.load_npz("files/icm.npz")

recommender = SLIM_Elastic_Net_Structured(ICM, URM_train)
recommender.fit(k=50, shrink=100, lamb=0.001, beta=0.001, gamma=0.0001, epochs= 50, normalize=True)
evaluation = recommender.evaluateRecommendations(URM_test)

```
### Results comparison: 

TheMoviesDataset (ICM sparcity: )

|              | KNN collaborative | KNN content base  | SLIM ElasticNet  | Structured SLIM  |
| -------------|:-----------------:| -----------------:| ----------------:| ----------------:|
|AUC           |0.16328739386752908|0.10482457612266392|0.2263805145071017|0.0712925984345927|
|MRR           |0.18905935960684533|0.11191219297558800|0.2645136270923570|0.0714445525835901|
|NDCG          |0.01855725520613782|0.00746201206025480|0.0311315445799227|0.0051277609005780|
|Map           |0.09102503054552336|0.03904454673930252|0.1272435077566548|0.0193769258401597|
|Precision     |0.12380611457273409|0.05838935934579915|0.1696978510621871|0.0329138280140035|
|Recall        |0.00925958646390481|0.00361115681867504|0.0149399014209997|0.0023711798147415|

Movielens26 with GENRES (ICM sparcity: )

|              | KNN collaborative | KNN content base  | SLIM ElasticNet  | Structured SLIM  |
| -------------|:-----------------:| -----------------:| ----------------:| ----------------:|
|AUC           |0.16531220636270120|0.02413698599489908|0.2271634077587352|0.0792913926051874|
|MRR           |0.19139018151447576|0.02351167837487133|0.2636167539188936|0.0787235073903345|
|NDCG          |0.01897790003872984|0.00110770278659267|0.0310036592400856|0.0050223485920076|
|Map           |0.09108725657623755|0.00595261532954509|0.1263267745014880|0.0206817543414812|
|Precision     |0.12431652422927689|0.01177681328023700|0.1680321267170991|0.0339791489552076|
|Recall        |0.00957347798587911|0.00061475823269545|0.0146979726141915|0.0021920817766374|

Movielens26 with TAGS (ICM sparcity: )

|              | KNN collaborative | KNN content base  | SLIM ElasticNet  | Structured SLIM  |
| -------------|:-----------------:| -----------------:| ----------------:| ----------------:|
|AUC           |0.16531220636270120|0.09306531239279914|0.2271456963025925|0.0320176517965009|
|MRR           |0.19139018151447576|0.09721575909436314|0.2636120930093823|0.0309206601339358|
|NDCG          |0.01897790003872984|0.00626021730383710|0.0310051664326851|0.0021933634099461|
|Map           |0.09108725657623755|0.03198887534117678|0.1263297574835752|0.0075111395737320|
|Precision     |0.12431652422927689|0.04987471475232986|0.1680433128999261|0.0148082688263470|
|Recall        |0.00957347798587911|0.00310033025947884|0.0146982335097261|0.0011481825420549|

### Authors: 
*Alex Porciani, Luca Vavassori*
