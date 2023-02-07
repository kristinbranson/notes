# Behavior Forecasting Notes

## GANet

Goal: Given past observed states for this actor $a_P = a_{-T':0}$, past states for other $N-1$ actors $a_P^O$, and a map of the environment $m$, estimate the distribution of future trajectories for this actor $a_F = a_{1:T}$:
$$p(a_F | m, a_P, a_P^O)$$
This can be broken up based on goals $\tau \in T(m,a_P,a_P^O)$ as
$$p(a_F | m, a_P, a_P^O) \approx \sum_{\tau \in T(m,a_P,a_P^O)} p(\tau | m, a_P, a_P^O) p(a_F | \tau, m, a_P, a_P^O)$$
GANet extracts features from the map around both the initial state $a_0$ and related to the goal area $\tau$:
$$p(a_F | m, a_P, a_P^O) \approx \sum_{\tau} p(\tau | M_{a_0}, a_P, a_P^O) p(M_\tau | m, \tau) p(a_F | \tau, M_\tau, M_{a_0}, a_P, a_P^O)$$

*Motion history encoding*: Following LaneGCN, they use a 1D CNN with Feature Pyramid Network (FPN) to extract motion features from actors' past states. 

*Goal prediction*: They train a residual MLP with two branches. One branch is trained to predict $E$ goal coordinates $G_{n,end} = \\{ g_{n,end}^e \\}$, and the second branch is trained to output $E$ confidence scores $C_{n,end} = \\{ c_{n,end}^e \\}$. The loss is the sum of a classification and regression loss. The classification loss is 
$$L_{clsend} = \frac{1}{N(E-1)} \sum_{n=0}^{N-1} \max(0,c_{n,end}^e + \epsilon - c_{n,end}^{\hat{e}})$$
where $\epsilon = 0.2$ is the margin and $\hat{e}$ is a positive goal with minimum Euclidean distance with the groundtruth trajectory's endpoint. 
The regression loss is
$$L_{regend} = \frac{1}{N} \sum_{n=0}^{N-1} \|g_{n,end}^{\hat{e}} - a_{n,end}^\*\| $$
where $a_{n,end}^\*$ is the true endpoint for actor $n$ for positive goals $\hat{e}$. They do something similar for a mid point $g_{n,mid}$. 
I don't understand where the negative goals are coming from. Are there in total $E$ discrete goals for all actors at all time points? 

*Motion estimation and scoring*: They train a two branch multi-modal prediction header to predict $K$ sequences of states $a_n^k$ for each actor $n$. The classification branch scores the trajectories:
$$L_{cls} = \frac{1}{N(K-1)} = \sum_{n=0}^{N-1} \sum_{k \neq \hat{k}} \max(0, c_n^k + \epsilon - c_n^{\hat{k}})$$
where $\hat{k}$ is the positive trajectory whose endpoint is closest to the true endpoint. 
The regression branch predicts the trajectories to minimize the L1 loss:
$$L_{reg} = \frac{1}{NT} \sum_{n=0}^{N-1} \sum_{t=1}^T \| a_{nt}^{\hat{k}} - a_{nt}^\* \|$$
There's additional weight placed on ending at the goal by including
$$L_{end} = \frac{1}{N} \sum_{n=0}^{N-1} \|(a_{n,end}^{\hat{k}} - a_{n,end}^\*\|$$

They train all the modules jointly. 


## References
1. [GANet: Goal Area Network for Motion Forecasting.](https://arxiv.org/abs/2209.09723). Top performer on Argoverse 2.0 motion forecasting task
2. [Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting](https://arxiv.org/pdf/2301.00493.pdf)
3. [Social LSTM: Human Trajectory Prediction in Crowded Spaces](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)
4. [SR-LSTM: State Refinement for LSTM Towards Pedestrian Trajectory Prediction](https://arxiv.org/abs/1903.02793)
5. [Unsupervised Trajectory Prediction with Graph Neural Networks](https://ieeexplore.ieee.org/document/8995232)
6. [Multi-Head Attention for Multi-Modal Joint Vehicle Motion Forecasting](https://ieeexplore.ieee.org/document/9197340)
7. [Multimodal Motion Prediction with Stacked Transformers](https://arxiv.org/abs/2103.11624)
8. [Desire: Distant Future Prediction in Dynamic Scenes with Interacting Agents](https://arxiv.org/abs/1704.04394)
9. [Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks](https://arxiv.org/abs/1803.10892)
10. [CoverNet: Multimodal Behavior Prediction Using Trajectory Sets](https://arxiv.org/abs/1911.10298)
11. [Map-Adaptive Goal-Based Trajectory Prediction](https://arxiv.org/abs/2009.04450)
12. [TNT: Target-Driven Trajectory Prediction](https://arxiv.org/abs/2008.08294)
13. [DenseTNT: End-to-End Trajectory Prediction from Dense Goal Sets](https://arxiv.org/abs/2108.09640)
14. [Learning Lane Graph Representations for Motion Forecasting](https://arxiv.org/abs/2007.13732)
