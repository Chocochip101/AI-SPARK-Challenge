## Baseline

### MSVA

- paper: https://arxiv.org/pdf/2104.11530.pdf
- code: https://github.com/TIBHannover/MSVA


### Clip

- paper: https://arxiv.org/pdf/2103.00020v1.pdf
- code: https://github.com/OpenAI/CLIP  


### Clip4clip
- paper: https://arxiv.org/pdf/2104.08860.pdf
- code: https://github.com/ArrowLuo/CLIP4Clip
<!-- 
- paper: https://arxiv.org/pdf/2105.12532.pdf
- code: https://github.com/TIBHannover/UnsupervisedVideoSummarization -->

### https://github.com/habla-liaa/ser-with-w2v2

## Video Feature Extraction

https://github.com/hobincar/pytorch-video-feature-extractor  
https://github.com/antoine77340/video_feature_extractor

## Benchmarks (for evaluation)

### (1) SumMe (Song et al., 2015)

- Download: https://gyglim.github.io/me/vsum/index.html#sf_code
- SumMe consists of 25 user videos covering various topics such as holidays and sports et cetera.
- Each video ranges from 1 to 6 minutes.
- SumMe is annotated by 15 to 18 persons, thus there are multiple ground truth summaries for each video.


### (2) TVSum (Song et al., 2015)

- Download: https://github.com/yalesong/tvsum
- TVSum contains 50 videos, which include the topics of news, documentaries, et cetera.
- The duration of each video varies from 2 to 10 minutes.
- Each video has 20 annotators that provide frame-level importance scores.


### (3) OVP (Open Video Project)

- Donwload: https://open-video.org/
- 50 videos


### (4) YouTube (De Avila et al., 2011)

- Download: 
- 39 videos
- Excluding cartoon videos


## Evaluation Metric

- F-score (protocol from Zhang et al. 2016: Video summarization with long short-term memory)

- https://arxiv.org/pdf/2109.06822.pdf

- https://arxiv.org/pdf/2007.14560.pdf


## Links

https://arxiv.org/pdf/2006.13979.pdf

https://arxiv.org/pdf/2201.02494.pdf

https://medhini.github.io/clip_it/

https://github.com/KaiyangZhou/pytorch-vsumm-reinforce

https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681

https://arxiv.org/pdf/2112.00007.pdf

https://velog.io/@jkl133/temperature-parameter-in-learner-fastai

https://arxiv.org/pdf/2109.14084.pdf

## Paper list (contrastive learning)

* NLP에서는

- CERT: Contrastive Self-supervised Learning for  Language Understanding, CORR 2020
- Supervised Contrastive Learning for Pre-trained Language Model Fine-Tuning, ICLR 2021
- FairFil: Contrastive Neural Debiasing Method for Pretrained Text Encoders, ICLR 2021
- Disentangled Contrastive Learning for Learning Robust Textual Representations, CICAI 2021
- SimCSE: Simple Contrastive Learning of Sentence Embeddings, EMNLP 2021
- Not All Negatives are Equal: Label-Aware Contrastive Loss for Fine-grained Text Classification, EMNLP 2021
- Self-Guided Contrastive Learning for BERT Sentence Representations, IJCNLP 2021
- DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations, IJCNLP 2021
Contrastive Learning for Fair Representation, AAAI 2022
- SimSCL : A Simple fully-Supervised Contrastive Learning Framework for Text Representation, AJCAI 2022


===============================

* VISION에서는 너무많은데 핵심논문은 아래 정도?

- [2] representation learning with contrastive predictive coding, arXiv 2018
- [8] Unsupervised Feature Learning via Non-Parametric Instance Discrimination, CVPR 2018
- [1] a simple framework for contrastive learning of visual representations, ICML 2020
- [3] understanding contrastive representation learning through alignment and uniformity on the hypersphere, ICML 2020
- [11] Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020
- [12] Supervised Contrastive Learning, NIPS 2020


================================

MULTIMODAL에서는 CLIP 이전 논문들 중에는 아래 논문 되게 괜찮고
- Cross-Modal Contrastive Learning for Text-to-Image Generation, CVPR 2021

- CLIP 이후로는 그냥 CLIP 인용논문내에서 형 관심 키워드 쳐서 같이 찾으면 많이나올듯!
