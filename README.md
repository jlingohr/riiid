# riiid


This repo contains code to train PyTorch models for the [Riiid AIEd Challenge 2020](https://www.kaggle.com/c/riiid-test-answer-prediction).

## Data Processing

Train-eval data splits are performs in the *riiid_splits.ipynb* notebook and were performed on a Google Colab instance with access.

## Supported Models

* **pebg**: PyTorch implementation of [*Improving Knowledge Tracing via Pre-training Question Embeddings*](https://www.ijcai.org/Proceedings/2020/0219.pdf) used to
pretrain question embeddings.
* **saint**: Pytorch implemen tation of [*SAINT+: Integrating Temporal Features for EdNet Correctness Prediction*](https://arxiv.org/abs/2010.12042) with pytorch lightning trainer.
