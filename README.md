# Boosting learning for LDPC codes

Implementation of boosting learning for LDPC codes described in ["Boosting Learning for LDPC Codes to Improve the Error-Floor Performance (NeurIPS 2023)"](https://arxiv.org/abs/2310.07194).

<p align="center">
<img src="BaseGraph/Block_Diagram.jpg" width="550px">
</p>

## Abstract

Low-density parity-check (LDPC) codes have been successfully commercialized
in communication systems due to their strong error correction ability and sim-
ple decoding process. However, the error-floor phenomenon of LDPC codes, in
which the error rate stops decreasing rapidly at a certain level, poses challenges in
achieving extremely low error rates and the application of LDPC codes in scenarios
demanding ultra high reliability. In this work, we propose training methods to
optimize neural min-sum (NMS) decoders that are robust to the error-floor. Firstly, 
by leveraging the boosting learning technique of ensemble networks, we divide the
decoding network into two networks and train the post network to be specialized
for uncorrected codewords that failed in the first network. Secondly, to address the
vanishing gradient issue in training, we introduce a block-wise training schedule
that locally trains a block of weights while retraining the preceding block. Lastly,
we show that assigning different weights to unsatisfied check nodes effectively low-
ers the error-floor with a minimal number of weights. By applying these training
methods to standard LDPC codes, we achieve the best error-floor performance com-
pared to other decoding methods. The proposed NMS decoder, optimized solely 
through novel training methods without additional modules, can be implemented
into current LDPC decoders without incurring extra hardware costs. The source
code is available at https://github.com/ghy1228/LDPC_Error_Floor.

## Implementation Enviroment

- NVIDIA RTX 2080Ti  
- Python = 3.6  
- Cudatoolkit=10.0  
- cudnn=7.6.5  
- tensorflow=2.4.0  

## Script

`python main.py

    
## License
This repo is MIT licensed.
