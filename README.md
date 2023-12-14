# SpectralNeRF: Physically Based Spectral Rendering with Neural Radiance Field
This is the Pytorch implementation of our AAAI 2024 paper SpectralNeRF.

![image](./figs/pipeline.png)

## Dependencies

* Pytorch
* Other requirements please refer to requirements.txt.

## Data Preparation

The datasets can be downloaded [here](https://drive.google.com/).

The dataset contains the synthetics datasets and real-world datasets

### Synthetic datasets

We render our synthetic scenes with multiple spectral illuminants to generate spectral images. To acquire spectral illuminants, we divide the wavelength range of the light source spectrum in Mitsuba from 360nm to 830nm into 11 adjacent intervals.

In addition, we use the CIE standard illuminant D65 as the default white light source for scenes in our dataset. The D65 light source is an artificial light source that simulates daylight, and its emission spectrum conforms to the average midday light of European and Pacific countries. 


### Real-world datasets

We utilize a camera and 8 color absorbers whose center wavelengths range from 400nm to 750nm with the interval of 50nm to capture the real-world scene. Different color absorbers are covered to the camera lens to obtain the spectral images.

## Results

[Link](https://htmlpreview.github.io/?https://github.com/liru0126/SpectralNeRF/blob/main/supp_videos/index.html)
              

## Acknowledgments

In this project we use (parts of) the implementations of the following work:

* [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

We thank the respective authors for open sourcing of their implementations.
