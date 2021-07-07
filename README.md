# UFOM

Code for the paper [Debiasing a First-order Heuristic for Approximate Bi-level Optimization](https://arxiv.org/abs/2106.02487) at ICML 2021. Note that this is mostly a fork off of the original [Reptile codebase](https://github.com/openai/supervised-reptile), but with a modified `reptile.py` file, which contains different MAML variants. The UFOM method can be found in the `UnbMAML` class.

Experiments with hyperparameter tuning configurations can be found in `commands.py`, while toy experiments can be found in the `toy` folder.

If you found this codebase to be useful, please consider citing our paper:

```
@article{ufom,
  author    = {Valerii Likhosherstov and
               Xingyou Song and
               Krzysztof Choromanski and
               Jared Davis and
               Adrian Weller},
  title     = {Debiasing a First-order Heuristic for Approximate Bi-level Optimization},
  journal   = {CoRR},
  volume    = {abs/2106.02487},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.02487},
  archivePrefix = {arXiv},
  eprint    = {2106.02487}
}
```
