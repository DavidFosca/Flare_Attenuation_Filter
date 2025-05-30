# Flare Attenuation Filter

Lens flare artifacts are undesired visual distortions caused by stray light, which can negatively impact the integrity and quality of an image. These artifacts pose a significant challenge in industrial applications like automotive and surveillance, where the quality and reliability of input images are crucial. Although there are techniques to prevent unwanted light from entering the camera, they are not always effective, requiring image post-processing methods.

In this work, a comprehensive review of image reconstruction algorithms based on deconvolution and stray light characterization through Point Spread Function (PSF) modeling is conducted. These approaches emphasized the significance of accurately modeling the stray light using the PSF specific to a camera system. However, necessary equipment to measure the PSF is unavailable for this work, and attempts to generate a synthetic PSF for evaluating the performance of a deconvolution algorithm resulted in unrealistic lens flare artifacts. Furthermore, literature studies primarily focused on static camera system setups like the ones found in microscopy or astronomy applications (telescope), indicating limited potential for PSF-based lens flare reduction in dynamic industrial settings.

On the other hand, artificial intelligence, particularly deep learning neural networks, have shown promising results in attenuating lens flare despite limited studies in this area. In this work, a synthetic flare dataset is generated, and an iterative training process that includes the evaluation of transfer learning is employed to develop FlareNet, the first compact and lightweight U-Net based model for lens flare reduction. FlareNet architecture, with a small parameter count of less than 150,000 parameters comprising convolutional layers, demonstrates improvement in image quality by reducing flare artifacts on synthetic test images. Furthermore, the model successfully reduces lens flare in real-life images, indicating its potential for achieving visually satisfactory results despite having less than 0.5\% of the weights of the state-of-the-art neural architecture used for this same application. Additionally, a quantization-aware approach is applied to assess the impact of reducing the weight representation from float32 to int8, resulting in a 30\% lighter model while considering the trade-off in accuracy.

This study serves as a proof-of-concept to understand the resource utilization and performance of implementing a model such as FlareNet, as a hardware-based digital circuit. To this end, the neural network is implemented in C++ using Vitis HLS, with each layer and necessary elements implemented and tested. Synthesis and validation are performed using the VITIS tool, and reports are analyzed while experimenting with HLS optimization directives. However, further work is necessary to optimize the overall design and explore more parallelism potential, making it feasible for deployment in real-time applications. Nevertheless, executing the model on a medium-end GPU demonstrates the possibility of meeting real-time requirements in terms of frames-per-second, but at the cost of higher power consumption, making it less suitable for low-power applications compared to an FPGA implementation.


<p align="center"> 
    <img src="https://github.com/DavidFosca/Flare_Attenuation_Filter/blob/main/flare_attenuation.png" alt="Resultado">
</p>

<p align="center"> 
    <img src="https://github.com/DavidFosca/Flare_Attenuation_Filter/blob/main/flare_attenuation2.png" alt="Resultado">
</p>

