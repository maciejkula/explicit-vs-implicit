
# Explicit vs implicit feedback models

This short notebook is meant to illustrate the importance of modelling implicit feedback in recommender systems. It reproduces the results of Harald Steck's [seminal paper](https://pdfs.semanticscholar.org/b7a6/4986251bfcf5fa3cac4e0c67ab2b2e78a082.pdf), _Training and Testing of Recommender Systems on Data Missing Not at Random_.

Its main thrust is this: it is _never_ appropriate the evaluate recommender system on observed ratings only, and when one evaluates a system based on constructing a ranking over all items, modelling implicit feedback is crucial.

For this experiment we're going to use [Spotlight](https://github.com/maciejkula/spotlight) to:

1. Fit an explicit recommender system based on observed ratings only.
2. Fit an implicit recommender system based on what ratings were and were not observed.
3. Compare their performance in ranking _all_ items using the [Mean Reciprocal Rank, (MRR)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) metric.

Let's import Spotlight first.


```python
import numpy as np

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.factorization import explicit, implicit
from spotlight.evaluation import mrr_score, rmse_score
```

Set some hyperparameters, get the dataset and split it into test and train.


```python
RANDOM_SEED = 42
LATENT_DIM = 32
NUM_EPOCHS = 10
BATCH_SIZE = 256
L2 = 1e-6
LEARNING_RATE = 1e-3

dataset = get_movielens_dataset('100K')
train, test = random_train_test_split(dataset, random_state=np.random.RandomState(RANDOM_SEED))
```

Create two models: and explicit feedback model trained to minimize the squared difference of true and predicted ratings on observed ratings only, and an implicit model whose goal is to rank all watched items over all items that weren't watched.


```python
explicit_model = explicit.ExplicitFactorizationModel(loss='regression',
                                                     embedding_dim=LATENT_DIM,
                                                     n_iter=NUM_EPOCHS,
                                                     learning_rate=LEARNING_RATE,
                                                     batch_size=BATCH_SIZE,
                                                     l2=L2,
                                                     random_state=np.random.RandomState(RANDOM_SEED))
implicit_model = implicit.ImplicitFactorizationModel(loss='bpr',
                                                     embedding_dim=LATENT_DIM,
                                                     n_iter=NUM_EPOCHS,
                                                     learning_rate=LEARNING_RATE,
                                                     batch_size=BATCH_SIZE,
                                                     l2=L2,
                                                     random_state=np.random.RandomState(RANDOM_SEED))
```

Fit the models (shouldn't take more than 30 seconds).


```python
explicit_model.fit(train)
implicit_model.fit(train)
```

Just to make sure that the explicit model is of decent quality, we compute its RMSE score on the test set. Anything below 1.0 is a reasonable model.


```python
print('Explicit RMSE: {:.2f}.'.format(rmse_score(explicit_model, test).mean()))
```

    Explicit RMSE: 0.94.


How good are the two models when trying to rank all items? A perfect score would rank all seen items in the test set over all unseen items (we exclude seen items from the training set from the evaluation), and return a score of 1.0.


```python
print('Explicit MRR: {:.2f}'.format(mrr_score(explicit_model, test, train=train).mean()))
```

    Explicit MRR: 0.02



```python
print('Implicit MRR: {:.2f}'.format(mrr_score(implicit_model, test, train=train).mean()))
```

    Implicit MRR: 0.07


The explicit model is _awful_: the two models are not even close. You should _never_ use pure explicit feedback models.
