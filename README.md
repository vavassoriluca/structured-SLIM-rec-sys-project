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

### Authors: 
*Alex Porciani, Luca Vavassori*
