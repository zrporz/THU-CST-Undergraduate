import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal

np.random.seed(2024) # DO NOT CHANGE THIS

def gen_data(n_sample, prior):
    n1 = int(n_sample * prior[0])
    n2 = n_sample - n1
    x_cls1 = np.random.multivariate_normal(mean=[-1,0], cov=0.5*np.eye(2), size=n1)
    x_cls2 = np.random.multivariate_normal(mean=[1,0], cov=0.5*np.eye(2), size=n2)
    return np.concatenate([x_cls1, x_cls2], axis=0)

prior_probability = [2/3, 1/3]
data = gen_data(1000, prior_probability)
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

### Initial Parameters, don't change
pi = [0.5, 0.5]
mu = [np.array([0, 1]), np.array([0, -1])]
sigma = [np.eye(2), np.eye(2)]

def compute_log_likelihood(data, pi, mu, sigma):
    n_data = len(data)
    log_likelihood = 0
    for i in range(n_data):
        likelihood = 0
        for j in range(len(pi)):
            likelihood += pi[j] * multivariate_normal.pdf(data[i], mu[j], sigma[j])
        log_likelihood += np.log(likelihood)
    return log_likelihood

############ YOUR ANSWER ############

def e_step(data, pi, mu, sigma):
    responsibilities = np.zeros((len(data), len(pi)))
    for i in range(len(data)):
        for j in range(len(pi)):
            responsibilities[i, j] = pi[j] * multivariate_normal.pdf(data[i], mu[j], sigma[j])
        responsibilities[i] /= np.sum(responsibilities[i])

    return responsibilities

def m_step(data, responsibilities):
    
    # Update mixture coefficients
    pi = np.mean(responsibilities, axis=0)
    # Update means
    mu = (responsibilities.T @ data) / responsibilities.sum(axis=0)[:, np.newaxis]
    
    # Update covariances
    n_components = responsibilities.shape[1]
    n_features = data.shape[1]
    sigma = np.zeros((n_components, n_features, n_features))
    
    for j in range(n_components):
        diff = data - mu[j]
        weighted_sum = np.einsum('i,ij,ik->jk', responsibilities[:, j], diff, diff) 
        sigma[j] = weighted_sum / responsibilities[:, j].sum()

    return pi, mu, sigma


# EM Algorithm
max_iterations = 200
log_likelihoods = []

for t in range(max_iterations):
    # E-step
    responsibilities = e_step(data, pi, mu, sigma)
    
    # M-step
    pi, mu, sigma = m_step(data, responsibilities)
    
    # Compute log-likelihood
    log_likelihood = compute_log_likelihood(data, pi, mu, sigma)
    log_likelihoods.append(log_likelihood)

# Plot log-likelihood
plt.plot(range(1, max_iterations + 1), log_likelihoods)
plt.xlabel('Update Steps')
plt.ylabel('Log-Likelihood')
plt.title('GMM的对数似然')
plt.show()

# Report final parameters after 200 update steps
print("200次更新后的最终参数：")
print("混合系数 (pi)：", pi)
print("均值 (mu)：", mu)
print("协方差 (sigma)：", sigma)
