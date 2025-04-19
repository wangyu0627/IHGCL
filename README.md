# [TKDE 2025] IHGCL: Intent-Guided Heterogeneous Graph Contrastive Learning for Recommendation
Lei Sang, Yu Wang, Yi Zhang, Yiwen Zhang* and Xindong Wu. 

This is the PyTorch implementation by <a href='https://github.com/wangyu0627'>@WangYu</a> for IHGCL model proposed in this [[paper]](https://ieeexplore.ieee.org/document/10857594)

We will organize the complete code and upload it after the paper is accepted for publication.

## Model Architecture
<img src='model_IHGCL.png' />

### Enviroments
- python==3.10
- pytorch==2.0
- cuda==118
- dgl==2.0
## How to Run the code
```
python main.py --dataset=amazon --device='cuda:0'
```
### 🔧 Optimal Hyperparameters for IHGCL on Different Datasets
The model parameters have not been saved for a long time. If there are any errors, please contact me via the issue.
| Dataset         | ssl_lambda | intra_lambda | ib_lambda | enc_num_layer | mask_rate | dec_num_layer | remask_rate | num_remasking | GCN_layer | epochs |
|-----------------|------------|--------------|-----------|---------------|-----------|---------------|-------------|---------------|-----------|--------|
| **Yelp**         | 0.05       | 0.005        | 0.001     | 1         | 0.2        | 1         | 0.2          | 2           | 3          | 50     |
| **LastFM**       | 0.02       | 0.005        | 0.001     | 1         | 0.1        | 1         | 0.3          | 1           | 2          | 150    |
| **Amazon**       | 0.02       | 0.001        | 0.001     | 1         | 0.0        | 1         | 0.1          | 1           | 2          | 80     |
| **Douban Book**  | 0.05       | 0.005        | 0.001     | 1         | 0.1        | 1         | 0.6          | 1           | 2          | 50     |
| **Douban Movie** | 0.05       | 0.005        | 0.001     | 1         | 0.4        | 1         | 0.8          | 1           | 2          | 80     |
| **Movielens**    | 0.05       | 0.010        | 0.001     | 1         | 0.3        | 1         | 0.6          | 1           | 2          | 200    |

## Citation

If you find this useful for your research, please kindly cite the following paper:

```bibtex
@article{2025IHGCL,
  title={Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation},
  author={Sang, Lei and Wang, Yu and Zhang, Yi and Zhang, Yiwen and Wu, Xindong},
  journal={IEEE Transactions on Knowledge and Data Engineering (TKDE)},
  year={2025},
  volume={37},
  number={4},
  pages={1915-1929}}
