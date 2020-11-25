

# Simulation

By Hao Gao, 2020/11/25

## LIF model

$$
\begin{align}
C\dot{V_i}&=-g_{L,i}(V_i-V_L)+I_{syn,i}+I_{ext,i}\quad V_i<V_{th,i}\\
V_i & = V_{rest} \qquad t\in[t_k^i,t_k^i+T_{ref}]
\end{align} \tag{1}
$$

where:

- $V_i$: Membrane potential
- $C$: Capacitor
- $g_{L,i}$: Leaky conductance
- $V_L$: Leaky reversal potential
- $I_{syn,i}$:  Synchronizing current
- $I_{ext,i}$: Input current
- $V_{th,i}$: Threshold voltage
- $V_{rest}$: Rest potential
- $t$: Time Stamp
- $t_k^i$: The kth spiking time of neuron $i0$
- $T_{ref}$:  Refractory period

## Synaptic Model

Considering AMPA, NAMPA, GABAa, GABAb as transmitters,
$$
\begin{align}
I_{syn,i}&=I_{AMPA,i}+I_{NAMPA,i}+I_{GABAa,i}+I_{GABAb,i}\\
I_{u,i}&=-g_{u,i}(V_i-V_i^u)J_{u,i}\\
\dot{J_{u,i}} &=-\frac{J_{u,i}}{\tau_i^u}+\sum_{k,j}w_{ij}\delta(t-t_k^i)
\end{align} \tag{2}
$$
where

- $u=AMPA, NAMPA, GABAa, GABAb$
- $g_{u,i}$: Transmitter conductance
- $V_i^u$: Transmitter reversal potential
- $J_{u,i}$: Transmitter concentration
- $w_{ij}$: Weights of connectivity

## Autoregressive model

Calcium signal model is described as 
$$
\begin{align}
c_{t,i}&=\sum_{q=1}^p\lambda_ic_{t-q,i}+s_{t,i}\\
y_{t,i}&=\alpha(c_{t,i}+b)+v_{t,i} \quad v_{t,i}\sim\mathcal{N}(0,\sigma_i^2)
\end{align} \tag{3}
$$
where:

- $c_{t,i}$: Calcium concentration
- $s_{t,i}$: Spike number
- $y_{t,i}$: Observed calcium
- $\alpha$: Amplitude
- $b$: Baseline
- $v_{t,i}$: White noise 

## Ensemble Kalman Filter (EnKF)

### Model

$$
\begin{align}
x_k&=f(x_{k-1}, u_k)+w_k\\
y_k&=h(x_k)+v_k
\end{align} \tag{4}
$$

where:

- $Q=E(w_k^Tw_k)$
- $R=E(v_k^Tv_k)$
- $P_{k|k}=cov(x_k-\hat{x}_{k|k})$
- $P_{k|k-1}=cov(x_k-\hat{x}_{k|k-1})$

### Sampling

Sampling $(x^1,x^2,...,x^N)=MultivariateNormal(x_0, P, N)$ 

where,

- $P$ is the initial error covariance matrix
- $N$ is the sampling number

### Predict

#### prior state estimation

$$
\begin{align}
x^i_{k|k-1}&=f(x^i_{k-1|k-1}, u_k)\\
\hat{x}_{k|k-1}&= \sum_{i=1}^{N}x^i_{k|k-1}\\
\end{align} \tag{5-1}
$$

#### prior covariance matrix estimation

error
$$
\begin{align}
e_{k|k-1} &= x_k-\hat{x}_{k|k-1}\\
&= f(x_{k-1}, u_k)+w_k-f(\hat{x}_{k-1|k-1}, u_k)\\
\end{align} \tag{5-2}
$$

$$
\begin{align}
P_{k|k-1}&=cov(e_{k|k-1})\\
&=cov(f(x_{k-1}, u_k)-f(\hat{x}_{k-1|k-1}, u_k)+w_k)\\
&=cov(f(x_{k-1}, u_k)-f(\hat{x}_{k-1|k-1}, u_k))+Q\\
&\approx \frac{1}{N-1}\sum_{i=1}^{N}cov(\hat{x}^i_{k|k-1}-\hat{x}_{k|k-1})+Q \\

\end{align} \tag{5-3}
$$

### Update

### Prior observed value

$$
\begin{align}
\hat{z}^i_{k|k-1}&=h(\hat{x}_{k|k-1}^i) \\
\hat{z}_{k|k-1}&=\frac{1}{N}\sum_{i=1}^{N}\hat{z}^i_{k|k-1}\\
\end{align} \tag{6-1}
$$

### System Uncertainty

Residual
$$
\begin{align}
y_{k|k-1}&=z_k-\hat{z}_{k|k-1}\\
&=h(x_{k-1})+v_k-h(\hat{x}_{k|k-1})
\end{align} \tag{6-2}
$$

$$
\begin{align}
S_k &= cov(y_{k|k-1})\\
&= cov(h(x_{k-1})+v_k-h(\hat{x}_{k|k-1}))\\
& = cov(h(x_{k-1})-h(\hat{x}_{k|k-1})) + R\\
&\approx \frac{1}{N-1}\sum_{i=1}^{N}cov(\hat{z}^i_{k|k-1}-\hat{z}_{k|k-1})+R\\
\end{align} \tag{6-3}
$$

### Cross covariance matrix
$$
\begin{align}
M_k &= cov(e_{k|k-1}, y_{k|k-1})\\
&\approx \frac{1}{N-1}\sum_{i=1}^{N}cov(\hat{x}^i_{k|k-1}-\hat{x}_{k|k-1}, \hat{z}^i_{k|k-1}-\hat{z}_{k|k-1})\\
\end{align} \tag{6-4}
$$

### Kalman gain
$$
\begin{align}
K_k&=M_kS_k^{-1}\\
\end{align} \tag{6-5}
$$

### Posterior estimation of x

Sample $(z^1,z^2,...,z^N)=MultivariateNormal(z, R, N)$ 
$$
\begin{align}
\hat{x}^i_{k|k}&=\hat{x}^i_{k|k-1}+K_k(z^i-\hat{z}^i_{k|k-1}) \\
\hat{x}_{k|k}&=\frac{1}{N}\sum_{i=1}^{N}\hat{x}^i_{k|k}\\
\end{align} \tag{6-6}
$$

### Posterior estimation of P

$$
\begin{align}
P_{k|k}&=cov(x_k-\hat{x}_{k|k})\\
&= cov(x_k-\hat{x}_{k|k-1}-K_k(z^i-\hat{z}^i_{k|k-1}))\\
&\approx P_{k|k-1}-K_kS_kK_k\\
\end{align} \tag{6-7}
$$


