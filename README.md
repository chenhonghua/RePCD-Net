# RePCD-Net: Feature-aware Recurrent Point Cloud Denoising Network

### Introduction
This repository is for our International Journal of Computer Vision (IJCV) 2022 paper '[RePCD-Net: Feature-aware Recurrent Point Cloud Denoising Network](https://link.springer.com/article/10.1007/s11263-021-01564-7)'. 

### Denoised result
We have released some denoised results in our work, please feel free to use them.

### Synthetic test dataset
We have aslo released our synthetic test dataset for a easiser comparison for future researchers. For the quantitative statistics, please refer to the table 2 in this paper. Note also that this dataset is built based on the '[PU-GAN](https://liruihui.github.io/publication/PU-GAN/)'.

### taining dataset
Download the training dataset `train_4000_normal_scale_label_weight_61_6.h5` from [here](https://drive.google.com/drive/folders/1xRWZ4eCGGdwQUOMZxIIC8EYZd4tDalWD?usp=sharing). Then put it in the folder `../h5_data`.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/chenhonghua/Re-PCD.git
   cd Re-PCD
   ```
2. Compile the TF operators
   Follow the above information to compile the TF operators. 

3. train the model:
   run:
   ```shell
   python main.py --phase train
   ```
   
4. Evaluate the model:
   run:
   ```shell
   cd codes
   python main.py --phase test
   ```
   You will see the input and output results in the folder `../data/test_data` and `../model/generator2_new6/result/`.
   
Note: During the test stage, we consider the entire input point cloud as a single entity. However, if the input point cloud contains a large number of points, it is advisable to partition it into smaller patches and process each patch individually as separate inputs.


### Citation
If you use this dataset, please consider citing our work.

@article{chen2022repcd,  
  title={RePCD-Net: Feature-Aware Recurrent Point Cloud Denoising Network},  
  author={Chen, Honghua and Wei, Zeyong and Li, Xianzhi and Xu, Yabin and Wei, Mingqiang and Wang, Jun},  
  journal={International Journal of Computer Vision},  
  pages={1--15},  
  year={2022},  
  publisher={Springer}  
}
