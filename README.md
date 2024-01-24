# DDPM Tensorflow & Keras


<p float="left">
  <img src="results/resultmain.png" width="600" height="400"> 
</p>


#### This work implements the Tensorflow & Keras version of DDPM based on official version written in tensorflow.

Diffusion models are a fascinating use of deep learning for simulating the gradual evolution of data over time. The concept behind generation using diffusion model is to iteratively transform a data point from a noise to a target distribution which is the original data distribution. This transformation is performed through a series of discrete steps. The training stage involves forward diffusion and backward diffusion.

In the forward diffusion process, we gradually add gaussian noise to the data for a random number of time steps 't' ('t' less than or equal to total number of predetermined time steps) resulting in a noised image. In the reverse diffusion process, we begin with the noised image (noised for 't' steps in forward diffusion) and we train a UNet model, which aids in lowering the noise level by predicting the noise added to the original image.

For generating new images, we simply perform reverse diffusion iteratively. To do this, we first sample noise from a standard gaussian and we predict and remove the noise gradually over the predetermined number of time steps which results in producing a image that resembles the original data distribution.


## Acknowledgements

 - [DDPM in pytorch](https://github.com/dome272/Diffusion-Models-pytorch)
 - [LHQ Dataset](https://universome.github.io/alis)
  - [CIFAR Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data)
 - [DDPM Official](https://github.com/hojonathanho/diffusion)



## Training & Generation demo

- [DDPM](https://www.kaggle.com/code/nithishm2410/diffusion-usage-v2)









