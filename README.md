# SpectralNeRF: Physically Based Spectral Rendering with Neural Radiance Field
This is the Pytorch implementation of our AAAI 2024 paper [SpectralNeRF](https://arxiv.org/pdf/2312.08692.pdf).

![image](./figs/pipeline.png)

## Installation

```
git clone https://github.com/liru0126/SpectralNeRF.git
cd SpectralNeRF
pip install -r requirements.txt
```

## Data Preparation

The datasets can be downloaded [here](https://drive.google.com/drive/folders/1fAnWkynYJ_w7PrNfxreIRyK6fTHVNgEA?usp=drive_link).

The detailed **render configurations for synthetic datasets** and **the capture instructions for real-world scenes** can be found [here](./datasets/dataset.md).


## Training
``` 
sh train.sh
```

## Testing

```
sh test.sh
```


## Citation

```
@inproceedings{li2024spectralnerf,
  title={SpectralNeRF: Physically Based Spectral Rendering with Neural Radiance Field},
  author={Li, Ru and Liu, Jia and Liu, Guanghui and Zhang, Shengping and Zeng, Bing and Liu, Shuaicheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3154--3162},
  year={2024}
}
```

## Acknowledgments

In this project we use (parts of) the implementations of the following work:

* [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

We thank the respective authors for open sourcing of their implementations.
