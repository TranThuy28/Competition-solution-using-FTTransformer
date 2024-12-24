

<h2>Child Mind Institute — Problematic Internet Use</h2>

  <p >
    Problematic Internet Uses is a competition hosted on Kaggle by Child Mind Institute based on The Healthy Brain Network (HBN) dataset.  The goal of this competition is to predict  children's Severity Impairment Index - a standard measure of problematic internet uses or SII in short based on the available data. 
    <a href="https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview">View Kaggle</a>
  </p>
</div>

<!-- BUILT WITH -->
## Built With

* [FT-Transformer]
* [MICE Imputation]
* [LightGBM]
* [XGBoost]
* [CatBoost]
* [TabNetRegressor]
* [KNN Impution]
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The competition data is compiled into two sources, parquet files containing the accelerometer (actigraphy) series and csv files containing the remaining tabular data. The majority of measures are missing for most participants. In particular, the target sii is missing for a portion of the participants in the training set. You may wish to apply non-supervised learning techniques to this data. The sii value is present for all instances in the test set.

Beside that, SII (label) is defined based on PCIAT test, which includes 20 questions, each question can receive value from 0 - 5 points. In our baseline, instead of making SII prediction, we decided to predict PCIAT total mark and map it to SII. 
<!-- DATA PROCESSING -->
## Data Processing

### Baseline
In baseline, we only use training normal tabular data (not include parquet). Additionally, we also drop some samples which has unreliable labels. SII comes from adding up scores from 20 questions in  PCIAT (Parent-Child Internet Addiction Test) , where each item is scored from 0 to 5. If even one question is missed from a record, we remove that entire record from our dataset. 

We do this because the SII needs all 20 item scores to be accurate - missing even one score could lead to an incorrect SII value. This is an example of potential wrong label.

<!-- MODEL -->
## Model

### Baseline
We use FTTransformer (FeatureTokenizer + Transformer) in our baseline

The architecture of FT-Transformer has three main components:

- **Input Transformation or “Feature Tokenizer”** :Converts raw tabular data into embeddings with an additional `[CLS]` token for processing and final prediction.
- **Transformer Layers**: Multiple Transformer layers process these embeddings through self-attention mechanisms, which allows the model to learn inter-feature dependencies
- **Output Prediction**: The final representation of the `[CLS]` (classification token) is used as the input for the prediction layer.This layer generates the final output, which could be a classification or regression result depending on the task.

![Ft architecture][ft-architecture]

 The reason for this choice is that Transformer with its attention mechanism has proposed novel performance on NLP and even Computer Vision (with Vision transformer). Therefore, we want to see its performance on tabular data, the field which dominated by  Gradient Boosted Decision Trees up to now.

 We trained the model with mentioned data 100 epochs and got 0.322 accuracy.
 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- IMPROVEMENT -->
## Improvement
### First improvement

In our baseline, we only fill null with mean, this method didn’t preserve correlations between features. Therefore, we tried MICE in our first improvement (Multivariate Imputation by Chained Equations) and get 0.331 score.

![first-improvement]

### Second Improvement

In this version, we process parquet file by using AutoEncoder (We took reference from top notebooks at that time), merge it to our processed CSV,  and use this merged data as new training data. Apart from that, we decided to drop seasons-related features because they has poor correlations with label and their distribution is nearly uniform.

![Second improvenment][second-improvement]
After all of this changing, we gain submission score increasing from 0.331 to 0.343. 

### Third improvement

Observing the log during training FTTransformer, we can see model got in overfitting state, so we decided to add batch norm layer in the model’s architecture. Apart from that, we use Grid Search CV to find the best hyperparamerters for the model with the following code. 
```python
paramm_grid = {
      'dim': [8, 16, 32],
      'depth': [2, 3],
      'heads': [2, 4],
      'learning_rate': [1e-2, 1e-3, 1e-4],
      'attn_dropout': [0.1, 0.2],
      'ff_dropout': [0.1, 0.2],
}
```

### Fouth improvement
We use bagging techniques, with 4 regressors including lightgbm, xgboost, catboost and FTTransformer with following weights. 
```python
voting_model = VotingRegressor(estimators=[
    ('lightgbm', LGBM_Model),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model),
    ('ftt', FTT_Model),
],weights=[4.0,4.0,5.0, 4.0])
```
This improvement made overall score increasing from 0.354 to 0.369.

### Final improvement

These gradient boosting model and our FT-Transformer is used in 3 phases ensemble training that made 3 submissions.
```python
voting_model = VotingRegressor(estimators=[
    ('lightgbm', LGBM_Model),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model),
    ('ftt', FTT_Model),
],weights=[4.0,4.0,5.0, 4.0])
```

```python
voting_model = VotingRegressor(estimators=[
    ('lightgbm', LGBM_Model),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model),
    ('ftt', FTT_Model),
    ('tabnet', TabNet_Model)
],weights=[4.0,4.0,4.0, 3.0,5.0])
```

```python
voting_model = VotingRegressor(estimators=[
    ('lightgbm', LGBM_Model),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model),
    ('tabnet', TabNet_Model)
],weights=[4.0,3.0,5.0, 4.0])
```

The final submission is the result of 3 submissions voting by MODE() function. Adding 3 times votingregressors made overall score increasing from 0.3 to 0.43.


<!-- CONTRIBUTORS -->
## Contributors

Nguyen Thi Lan Huong - 22028151@vnu.eu.vn

Nguyen Minh Dung - 22028125@vnu.eu.vn

Tran Thi Thuy - 22028302@vnu.eu.vn

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Our code is based on [FT-Transformer] and some [top notebooks from Kaggle]. We also thank to [Kaggle] for organizing this competition and sharing datasets.




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[FT-Transformer]: https://arxiv.org/pdf/2106.11959
[MICE Imputation]: https://jeffgill.org/wp-content/uploads/2021/04/mice_multivariate_imputation_by_chained_equations.pdf
[LightGBM]: https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf
[XGBoost]: https://arxiv.org/abs/1603.02754
[CatBoost]: https://arxiv.org/abs/1706.09516
[TabNetRegressor]: https://github.com/dreamquark-ai/tabnet
[KNN Impution]: https://www.researchgate.net/publication/220981745_A_Study_of_K-Nearest_Neighbour_as_an_Imputation_Method
[ft-architecture]: https://github.com/TranThuy28/INT3405E-55--GROUP-5/blob/main/images/fttransformer.png
[first-improvement]: https://github.com/TranThuy28/INT3405E-55--GROUP-5/blob/main/images/firstimprovement.png
[second-improvement]: https://github.com/TranThuy28/INT3405E-55--GROUP-5/blob/main/images/secondimprovement.png
[top notebooks from Kaggle]: https://www.kaggle.com/code/kleedg/cmi-piu-lee-dong-gi#16.-Submission3
[Kaggle]: https://www.kaggle.com/
