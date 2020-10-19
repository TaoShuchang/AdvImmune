
# AdvImmune
This repository is an implementation of our proposed CoupledGNN model in the following paper:

```
Shuchang Tao, Huawei Shen, Qi Cao, Liang Hou, Xueqi Cheng. 2021. Adversarial Immunization
for Improving Certifiable Robustness on Graphs. In WSDM'21, Marck 8-12, 2021, Jerusalem, Israel, 9 pages.
​```
```

AdvImmune vaccinate an affordable fraction of node pairs, connected or unconnected, to improve the certifiable robustness of the graph against any admissible adversarial attack.

For more details, you can download this paper [Here](https://arxiv.org/abs/2007.09647)

## Requirements

Python 3.6

Pytorch 1.1.0

## Usage
***Example Usage***

`python -u main.py --dataset=citeseer `

For detailed description of all parameters, you can run

`python -u main.py --help`

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{tao2021advimmune,
  title={Adversarial Immunization for Improving Certifiable Robustness on Graphs},
  author={Shuchang Tao, Huawei Shen, Qi Cao and Liang Hou and Xueqi Cheng.},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  series={WSDM'21},
  year={2021},
  location={Jerusalem, Israel},
  numpages={9}
}
```