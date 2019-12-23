# Sparse Signal Recovery via Generalized Entropy Functions Minimization
* Author: Shuai Huang, The Johns Hopkins University.
* Email: shuai.huang@emory.edu

```
Last change: 12/29/2018
Change log: 
    v1.0 (SH) - (03/30/2017)
    v1.1 (SH) - (01/24/2018)
    v2.0 (SH) - (12/29/2018)
    v2.1 (SH) - (12/23/2019) update the image recovery algorithm with faster implementation that does not rely on alternating the split bregman shrinkage algorithm
```  


* This package contains source code for performing sparse signal recovery via generalized entropy function minimization approach described in the following paper:
```
@ARTICLE{GEFM17,
    author={S. Huang and T. D. Tran},
    journal={IEEE Transactions on Signal Processing},
    title={Sparse Signal Recovery via Generalized Entropy Functions Minimization},
    year={2019},
    volume={67},
    number={5},
    pages={1322-1337}, 
    month={March},
}
```
If you use this code and find it helpful, please cite the above paper. Thanks :smile:

## Summary
The package is written in MATLAB:
```
    1) The folder "code" contains the functions to perform sparse signal recovery using different approaches: \|x\|_1 minimization, \|\vx\|_p^p  minimization, generalized shannon entropy function minimization, generalized renyi entropy function minimization, L_1/L_infinity, logarithm of energy, iterative hard thresholding, OMP, CoSaMP
    2) The folder "sara_weight" contains the funtions to perform image recovery from linear measurements
    3) "noiseless_signal_recovery.m" contains examples to perform noiseless recovery.
    4) "noisy_signal_recovery.m" contains examples to perform noisy recovery.
    5) "image_recovery.m" contains examples to perform image recovery via sparsity averaging.
    6) "face_recognition.m" contains examples to use SRC to perform face recognition.
```
