# Official PyTorch Implementation of NeuCGC

**Liang Peng, Yixuan Ye, Cheng Liu\* , Hangjun Che, Man-Fai Leung, Si Wu, and Hau-San Wong. "Trustworthy Neighborhoods Mining: Homophily-Aware Neutral Contrastive Learning for Graph Clustering" (TKDE Submission)**



## Abstract

Recently, neighbor-based contrastive learning has been introduced, effectively leveraging neighborhood information to enhance the clustering process. However, these methods often rely on the homophily assumption, which posits that connected nodes are likely to belong to the same class and should therefore be close in feature space. This assumption overlooks the varying levels of homophily present in real-world graphs. As a result, applying contrastive learning to graphs with low homophily can lead to indistinguishable node representations due to the incorporation of unreliable neighborhood information. Consequently, identifying trustworthy neighborhoods with varying homophily levels poses a significant challenge in contrastive graph clustering. To tackle this, we introduce a novel neighborhood contrastive graph clustering method that extends traditional neighborhood contrastive learning by incorporating neutral pairs—pairs of nodes treated as weighted positive pairs, rather than strictly positive or negative. These neutral pairs are dynamically adjusted based on the graph’s homophily level, enabling a more flexible and robust learning process. Leveraging neutral pairs in contrastive learning, our method incorporates two key components: (1) an adaptive contrastive neighborhood distribution alignment that adjusts based on the homophily level of the given attribute graph, ensuring effective alignment of neighborhood distributions, and (2) a contrastive neighborhood node feature consistency learning mechanism that leverages reliable neighborhood information from high-confidence graphs to learn robust node representations, mitigating the adverse effects of varying homophily levels and effectively exploiting highly trustworthy neighborhood information. Experimental results demonstrate the effectiveness and robustness of our approach, outperforming other state-of-the-art graph clustering methods.



## Requirements

- CUDA Version: 11.6
- numpy==1.26.1
- torch==1.12.1+cu116
- tqdm==4.66.1
- logging==0.5.1.2



## Datasets

Datasets (Cora, Citeseer, Pubmed, DBLP, ACM, Photo, Wisconsin, Cornell, Texas) are publicly available [here](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/blob/main/dataset/README.md).

Datasets (Chameleon and Crocodile) are publicly available [here](https://graphmining.ai/datasets/ptg/wiki/).



## Usage

To replicate the experimental results in the paper, run the following script

```
python run.py
```



## Acknowledgments

Our code is inspired by [Awesome-Deep-Graph-Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering).









