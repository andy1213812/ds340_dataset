# Dataset structure

POLIDriving contains driving data from 5 drivers (alonso, andres, pablo, richard, and yolanda) and also synthetic data from one unreal driver (furious).  

The folder structure of POLIDriving is the following.

```
dataset
|
+-- alonso
|   |
|   +-- 20231229_151643
|       |
|       +-- preview.png (first frame video)
|       +-- raw_log.bz2 (raw capnp log, can be read with openpilot-tools: logreader)
|       +-- video.hevc (video file, can be read with openpilot-tools: framereader)
|       +-- processed_log/ (processed logs as numpy arrays, see format for details)
|   +-- global_pos/ (global poses of camera as numpy arrays, see format for details)
+-- alonso
    |
    +-- preview.png (first frame video)
    +-- raw_log.bz2 (raw capnp log, can be read with openpilot-tools: logreader)
    +-- video.hevc (video file, can be read with openpilot-tools: framereader)
    +-- processed_log/ (processed logs as numpy arrays, see format for details)
    +-- global_pos/ (global poses of camera as numpy arrays, see format for details)

```

# Publication

If you use POLIDriving in your research, please cite it as follows.

@article{marcillo2024polidriving,  
title={ public-access driving dataset for road traffic safety analysis},  
author={Marcillo, Pablo and Arciniegas-Ayala, Cristian and Valdivieso Caraguay, Ángel Leonardo and Sanchez-Gordon, Sandra and Hernández-Álvarez, Myriam},  
journal={arXiv preprint arXiv:},  
year={2024}  
}

# Downloads

The size of POLIDriving is about 20 MB.

# Contact

For questions or suggestions, please contact pablo.marcillo@epn.edu.ec
