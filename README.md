# News Categorizer

News categorizer is docker-based web service:

  - provides category of given news url
  - provides category of given news body

### Installation
Edit docker-compose.yml based on your server. Then run the following command:
```sh
docker-compose up
```

### Dataset
The dataset we use for the predictive model is [BBC News](https://www.kaggle.com/c/learn-ai-bbc/data). We split ``BBC News Train.csv`` into %20 of the data as validation, %10 of the data test, and the rest as train set by using random seed 42.

### Technologies
It uses [BERT](https://arxiv.org/abs/1810.04805) to predict the category given content. BERT has fine tuned by using [ktrain](https://github.com/amaiya/ktrain) library in [Colab](https://colab.research.google.com/drive/1NjjO7oGoKtXuPKSsFgLRW_1z_fd5r4mb). For url based detection, it uses rule-based approach. 

License
----

MIT

