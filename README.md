
# Adversarial Immunization
This repository is an implementation of the following paper:

**[Adversarial Immunization
for Certifiable Robustness on Graphs](https://arxiv.org/abs/2007.09647)**

By Shuchang Tao, Huawei Shen, Qi Cao, Liang Hou and Xueqi Cheng

Published at WSDM'21, March 2021 (virtual event)

*Adversarial immunization* vaccinate an affordable fraction of node pairs, connected or unconnected, to improve the certifiable robustness of the graph against any admissible adversarial attack.



## Requirements

- pytorch 
- scipy
- numpy
- numba
- cvxpy





## Usage
***Example Usage***

`python -u main.py --dataset=citeseer `

For detailed description of all parameters, you can run

`python -u main.py --help`



## 



## Cite

If you would like to use our code, please cite:
```
@inproceedings{tao2021advimmune,
  title={Adversarial Immunization for Certifiable Robustness on Graphs},
  author={Shuchang Tao, Huawei Shen, Qi Cao and Liang Hou and Xueqi Cheng.},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  series={WSDM'21},
  year={2021},
  location={Jerusalem, Israel},
  numpages={9}
}
```
