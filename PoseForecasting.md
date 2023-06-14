# Pose Forecasting Notes

## [Best Practices for 2-Body Pose Forecasting](https://arxiv.org/pdf/2304.05758.pdf)

![image](https://github.com/kristinbranson/notes/assets/211380/cdbc3dda-8e0e-4a04-9d22-2ffd997e4054)

From a sequence of $T$ frames or 2 humans each with $J$ joints, they want to predict the pose of both humans at frames $\{T+1,...,T+N}$. 

This paper evaluates various methods and compositions of methods proposed in previous papers for pose forecasting. They break up the algorithm into the input representation, encoding, and decoding stages.

They find better performance by representing the input and output pose sequences based on movement frequencies via the Discrete Cosine Transform (DCT). This was proposed in ["Learning Trajectory Dependencies for Human Motion Prediction"](https://arxiv.org/pdf/1908.05436.pdf). 
For each joint, the $i$th coefficient is:
$$F_{i}(X_j) = c_1 \sum_{t=1}^T x_{jt} c_2(i) cos( \pi(2t-1)(i-1)/2T )$$
where $X_j \in \mathbb{R}^{T \times 3}$ is the input joint position sequence, $x_{jt} \in \mathbb{R}^3$ is the joint position at time $t$, and $c_1$ and $c_2(i)$ are multipliers:
$$c_1 = \sqrt{2/T}$$
$$c_2(i) = (1+I(i=1))^{-1/2}$$
$i$ ranges from $1$ to $T$, with $i=1$ being high frequency and $i=T$ being low frequency. 
Before the final output, they also include the inverse DCT. 

They also find better performance by learning the adjacency matrix of a factorized space-time Graph Convolutional Network (GCN). This was proposed in ["Space-Time-Separable Graph Convolutional Network for Pose Forecasting"](https://arxiv.org/abs/2110.04573). The output of layer $l$ is:
$$H_{tvc}^{(l)} = \sum_{v',t',c'} A_{vv'}^{s(l)} A_{tt'}^{t(l)} H_{t'v'c'}^{(l-1)} W_{cc'}^{(l)}$$
where $A^s \in \mathbb{R}^{V \times V}$ is the learned spatial adjacency matrix (values between -1 and 1),
$A^t \in \mathbb{R}^{T \times T}$ is the learned temporal adjacency matrix (values between -1 and 1),
$W$ is the learned weight matrix. 
In ["Space-Time-Separable Graph Convolutional Network for Pose Forecasting"](https://arxiv.org/abs/2110.04573), they use 4 layers of GCNs with $C^{(1)} = 64$, $C^{(2)} = 32$, $C^{(3)} = 64$, and $C^{(4)} = 3$, . That's quite few layers!
In ["Back to MLP: A Simple Baseline for Human Motion Prediction"](https://arxiv.org/pdf/2207.01567.pdf), the input sequence length is 50 and the output length is 10 or 25 frames, depending on the data set. 

From the matrix $H^{(L)} \in \mathbb{R}^{T \times 2J \times C}$, they find that a fully-connected decoder to the output matrix of size $N \times 2J \times 3$. 

They also consider graph attention networks, unfactored graph adjacency matrices, and fixed spatial adjacency matrix based on the kinematic tree. This table shows the improvement of each of these best practices:
![image](https://github.com/kristinbranson/notes/assets/211380/6d9314dd-0028-49bd-9d63-691ff4ed977a)
 
