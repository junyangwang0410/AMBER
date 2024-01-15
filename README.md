# AMBER: An Automated Multi-dimensional Benchmark for Multi-modal  Hallucination Evaluation
<div align="center">
Junyang Wang*<sup>1</sup>, Yuhang Wang*<sup>1</sup>, Guohai Xu<sup>2</sup>, Jing Zhang<sup>1</sup>, Yukai Gu<sup>1</sup>, Haitao jia<sup>1</sup>, Jiaqi Wang<sup>1</sup>
</div>
<div align="center">
Haiyang Xu<sup>2</sup>, Ming Yan<sup>2</sup>, Ji Zhang<sup>2</sup>, Jitao Sang<sup>1</sup>
</div>
<div align="center">
<sup>1</sup>Beijing Jiaotong University    <sup>2</sup>Alibaba Group
</div>
<div align="center">
*Equal Contribution
</div>
<div align="center">
    <a href="https://arxiv.org/abs/2311.07397"><img src="README_File/Paper-Arxiv-orange.svg" ></a>
</div>

## Introduction
*AMBER* is **A**n LLM-free **M**ulti-dimensional **Be**nchma**r**k for MLLMs hallucination evaluation, which can be used to evaluate both generative task and discriminative task including existence, attribute and relation hallucination.
![](README_File/comparison.jpg?v=1&type=image)
*AMBER* has a fine-grained annotation and automated evaluation pipeline.
![](README_File/intro.jpg?v=1&type=image)
The data statistics and objects distribution.
![](README_File/statistics.jpg?v=1&type=image)
The results of mainstream MLLMs evaluated by AMBER.
![](README_File/result.jpg?v=1&type=image)

## News
* 🔥 [11.17] Our data and annotations are available!
* [11.14] Our paper is available at [LINK](https://arxiv.org/abs/2311.07397).

## Getting Started

### Installation

**1. spacy** is used for near-synonym judgment
```bash
pip install -U spacy
python -m spacy download en_core_web_lg
```
**2. nltk** is used for objects extraction
```bash
pip install nltk
```

### Image Download

Download the images from this [LINK](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing).

### Responses Generation
|  json file   | Task or Dimension   | Evaluation args | 
|:-------:|:-------:|:-------:|
|query_all.json| All the tasks and dimensions | a |
|query_generative.json| Generative task | g |
|query_discriminative.json| Discriminative task | d |
|query_discriminative-existence.json| Existence dimension | de |
|query_discriminative-attribute.json| Attribute dimension | da |
|query_discriminative-relation.json| Relation dimension | dr |

For generative task (1 <= id <= 1004), the format of responses is:
```bash
[
	{
		"id": 1,
		"response": "The description of AMBER_1.jpg from MLLM."
	},
	
	......
	
	{
		"id": 1004,
		"response": "The description of AMBER_1004.jpg from MLLM."
	}
]
```

For discriminative task (id >= 1005), the format of responses is:
```bash
[
	{
		"id": 1005,
		"response": "Yes" or "No"
	},
	
	......
	
	{
		"id": 15220,
		"response": "Yes" or "No"
	}
]
```

### Evaluation
```bash
python inference.py --inference_data path/to/your/inference/file --evaluation_type {Evaluation args}
```

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@article{wang2023llm,
  title={An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation},
  author={Wang, Junyang and Wang, Yuhang and Xu, Guohai and Zhang, Jing and Gu, Yukai and Jia, Haitao and Yan, Ming and Zhang, Ji and Sang, Jitao},
  journal={arXiv preprint arXiv:2311.07397},
  year={2023}
}
```
