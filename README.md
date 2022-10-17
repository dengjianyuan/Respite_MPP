### Taking a Respite from Representation Learning for Molecular Property Prediction

In the past decade, the practice of drug discovery has been undergoing radical transformations in light of the rapid development of artificial intelligence (AI). Among them, one major task is representation learning for molecular property prediction. 
Despite the ever-growing technical endeavor, a respite is needed at this point to rethink the key aspects underlying molecular property prediction.

---
This is the repository for the manuscript: [Taking a Respite from Representation Learning for Molecular Property Prediction](https://web10.arxiv.org/abs/2209.13492) by Jianyuan Deng et al.
In this study, we have conducted a systematic comparison of the representative models, namely, random forest, MolBERT and GROVER, which utilize three major molecular representations, 1) extended-connectivity fingerprints, 2) SMILES strings and 3) molecular graphs. 
Based on extensive experiments and evaluation, our central thesis is that **_"a model cannot save an unqualified dataset which cannot remedy an improper evaluation for an ambiguous chemical space generalization claim"_**. 

<p align="center">
  <img width="820" height="280" src="/images/respite22.png">
</p>

This repository provides data and results together with codes used for data processing and results analysis.


**Note** <br>
* MolBERT is based on the manuscript **Molecular representation learning with language models and domain-relevant auxiliary tasks** ([Paper](https://arxiv.org/abs/2011.13230); [Code](https://github.com/BenevolentAI/MolBERT)) by Benedek Fabian et al. <br>
* GROVER is based on the manuscript **Self-Supervised Graph Transformer on Large-Scale Molecular Data** ([Paper](https://arxiv.org/abs/2007.02835); [Code](https://github.com/tencent-ailab/grover)) by Yu Rong et al.