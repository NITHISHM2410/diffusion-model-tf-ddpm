# Exploring Landscape Image Generation Through Diffusion Model
Diffusion models are a fascinating use of deep learning for simulating the gradual evolution of data over time. 
The concept behind generation using diffusion model is to iteratively transform a data point from a noise to a target distribution which is the original data distribution. This transformation is performed through a series of discrete steps. The training stage involves forward diffusion and backward diffusion.  
<br>
In the forward diffusion process, we gradually add gaussian noise to the data for a random number of time steps 't' ('t' less than or equal to total number of predetermined time steps) resulting in a noised image. In the reverse diffusion process, we begin with the noised image (noised for 't' steps in forward diffusion) and we train a UNet model, which aids in lowering the noise level by predicting the noise added to the original image.


For Generating new images, we simply perform reverse diffusion iteratively. To do this, we first sample noise from a standard gaussian and we predict and remove the noise gradually over the predetermined number of time steps which results in producing a image that resembles the original data distribution.

  

**Dataset used**  
i) https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images

  
**Results**  

<p float="left">
  <img src="results/results10.png" width="300" ">
  <img src="results/results6.png" width="300" > 
  <img src="results/results11.png" width="300" >
</p>

  
**References**  
i) https://github.com/dome272/Diffusion-Models-pytorch

