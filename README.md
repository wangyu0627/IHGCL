# IHGCL: Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation
Lei Sang, Yu Wang, Yi Zhang, Yiwen Zhang* and Xindong Wu. 

This is the PyTorch implementation by <a href='https://github.com/wangyu0627'>@WangYu</a> for IHGCL model proposed in this paper[https://ieeexplore.ieee.org/document/10857594][TKDE 2025]

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
## Citation

If you find this useful for your research, please kindly cite the following paper:

```bibtex
@article{2025IHGCL,
  author={Sang, Lei and Wang, Yu and Zhang, Yi and Zhang, Yiwen and Wu, Xindong},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation}, 
  year={2025},
  pages={1-14},
  doi={10.1109/TKDE.2025.3536096}}
