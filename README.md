### Taking a Respite from Representation Learning for Molecular Property Prediction

In the past decade, the practice of drug discovery has been undergoing radical transformations in light of the rapid development of artificial intelligence (AI). Among them, one major task is representation learning for molecular property prediction. 
Despite the ever-growing technical endeavor, a respite is needed at this point to rethink the key elements underlying molecular property prediction.

--- 
This is the repository for the preprint: [Taking a Respite from Representation Learning for Molecular Property Prediction](https://arxiv.org/abs/2209.13492)) by Jianyuan Deng et al.
In this study, we have conducted a systematic evaluation on molecular property prediction. 
Based on extensive experiments and evaluation, our central thesis is that **_"a model cannot save an unqualified dataset which cannot remedy an improper evaluation for an ambiguous chemical space generalization claim"_**. 

<p align="center">
  <img width="800" height="200" src="/images/respite22.png">
</p>

This repository provides code (directory `src`), data (directory `data`) and results (directory `results`) together with notebooks (directory `notebooks`) used for data processing and results analysis. (Due to size limit, the exact data split files and raw prediction results are not attached in the repository, which are available upon request.)


**Note** <br>
* MolBERT is based on the manuscript **Molecular representation learning with language models and domain-relevant auxiliary tasks** ([Paper](https://arxiv.org/abs/2011.13230); [Code](https://github.com/BenevolentAI/MolBERT)) by Benedek Fabian et al. <br>
* GROVER is based on the manuscript **Self-Supervised Graph Transformer on Large-Scale Molecular Data** ([Paper](https://arxiv.org/abs/2007.02835); [Code](https://github.com/tencent-ailab/grover)) by Yu Rong et al.