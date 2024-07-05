from torchvision import datasets, transforms
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import loguru
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

import os
def plot_pca(vectors,output_path,label_list,n_components=2):
    from sklearn.decomposition import PCA

    # 使用PCA进行降维
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(vectors)

    # 绘制PCA降维结果，并调整标签位置
    plt.figure(figsize=(20, 16))
    plt.scatter(pca_result[:, 0], pca_result[:, 1],color=label_list,linewidths=10)
    # for i, word in enumerate(words):
    #     plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]), xytext=(5, 2), textcoords='offset points', fontsize=10)  # 设置标签的偏移量
    plt.title('PCA Word Embedding Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(output_path)

def plot_tsne(vectors,centroids,output_path, clusters,indices,labels,label_color_dict,n_components=2):
    
    k = centroids.shape[0]
    label_list = [label_color_dict[label] for label in labels[indices]]
    center_labels = []
    center_colors = []
    for i in tqdm(range(k)):
        cluster_indices = np.where(clusters == i)[0]
        true_labels = labels[cluster_indices]
        _index = np.bincount(true_labels).argmax() if len(true_labels) > 0 else np.random.randint(0, k)
        center_labels.append(_index)
        center_colors.append(label_color_dict[_index])
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=n_components, perplexity=30, n_iter=1000,random_state=random_seed)
    vectors = vectors[indices]
    tsne_result = tsne.fit_transform(np.concatenate((vectors,centroids)))

    # 绘制t-SNE降维结果
    plt.figure(figsize=(20, 16))
    plt.scatter(tsne_result[:-k, 0], tsne_result[:-k, 1],c=label_list,linewidths=5,alpha=0.5,marker='o')
    plt.scatter(tsne_result[-k:, 0], tsne_result[-k:, 1],c=center_colors,linewidths=50,marker='x')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(output_path)

def get_dataset(root_dir='data',compressed_dim=50,compress_method="pca"):
    train_data = datasets.MNIST(root=root_dir, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    images, labels = next(iter(train_loader))
    images = images.numpy().reshape(images.shape[0], -1)
    if compress_method == "pca":
        pca = PCA(n_components=compressed_dim,random_state=random_seed)
        compressed_images = pca.fit_transform(images)
    elif compress_method == "random":
        random_indices = np.random.choice(images.shape[1], compressed_dim, replace=False)
        compressed_images = images[:, random_indices]
    elif compress_method == "no_compress":
        compressed_images = images
    else:
        raise ValueError("Unsupported compress method")
    return compressed_images, labels.numpy()
def distance(a, b,mode="l2"):
    if mode == "l2":
        return np.linalg.norm(a - b,ord=2)
    elif mode == "l1":
        return np.linalg.norm(a - b,ord=1)
    elif mode == "linf":
        return np.linalg.norm(a - b,ord=np.inf)
    elif mode == "cosine":
        return -np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
def kmeans(data, k, max_iterations=100,eval_freq=2,init='random',distance_mode='l2',save_dir="default_log"):
    images, labels = data
    if init == 'random':
        indices = np.random.choice(len(images), k, replace=False)
    elif init == "farest":
        indices = []
        for i in tqdm(range(k)):
            if i == 0:
                indices.append(np.random.choice(len(images)))
            else:
                distances_list = []
                # random_indice = np.random.choice(len(images),1000)
                random_indice = range(len(images))
                for i, image in tqdm(enumerate(images[random_indice])):
                    # import pdb; pdb.set_trace()
                    distances_list.append(sum([distance(image, images[c],mode=distance_mode) for c in indices]) )
                indices.append(random_indice[np.argmax(distances_list)])
    else:
        raise ValueError("Unsupported init method")
    logger.info("Kmeans init done!")
    centroids = images[indices]
    for iter_ in tqdm(range(max_iterations)):
        old_centroids = centroids.copy()
        clusters =[]
        for point in images:
            distances = [distance(point, centroid,mode=distance_mode) for centroid in centroids]
            clusters.append(np.argmin(distances))
        clusters = np.array(clusters)
        centroids = []
        for cluster in range(k):
            centroid = np.mean(images[clusters == cluster], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        # 如果聚类中心不再改变，则停止迭代
        if np.all(old_centroids == centroids):
            break
        if (iter_) % eval_freq == 0:
            # 计算当前聚类结果的准确率
            accuracy = eval(clusters, labels, k)
            logger.info(f"Iter:{iter_},Accuracy:{accuracy:.4f}")
            randome_num = 2000
            random_indices = np.random.choice(len(images), randome_num, replace=False)
            label_color_dict = generate_color_map(labels[random_indices])
            # import pdb; pdb.set_trace()
            plot_tsne(images,centroids,os.path.join(save_dir,f"iter{iter_}.png"),clusters,random_indices,labels,label_color_dict)
            
    return clusters, centroids

def eval(clusters, labels, k):
    # 为每个聚类选择最常见的真实标签作为预测标签
    predicted_labels = np.zeros_like(clusters)
    
    for i in tqdm(range(k)):
        cluster_indices = np.where(clusters == i)[0]
        true_labels = labels[cluster_indices]
        predicted_labels[cluster_indices] = np.bincount(true_labels).argmax()
    accuracy = accuracy_score(labels, predicted_labels)
    return accuracy


def elbow_k(data,max_k):
    images, labels = data
    cost_values = []
    for k in tqdm(range(1, max_k,5)):
        save_dir = os.path.join("log","elbow_k")
        os.makedirs(save_dir,exist_ok=True)
        clusters, centroids = kmeans(data, k, max_iterations=100, eval_freq=100,save_dir=save_dir)
        wcss = 0
        for i in range(k):
            cluster_points = images[clusters == i]
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
        
        cost_values.append(wcss)
    plt.plot(range(1, max_k,5), cost_values, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(range(1, max_k,5))
    plt.savefig("elbow_k.png")

def generate_color_map(labels):
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    color_map = plt.cm.get_cmap('tab20', num_labels)
    label_color_dict = {label: color_map(i) for i, label in enumerate(unique_labels)}
    return label_color_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,help="estimate_k or kmeans",default="kmeans")
    parser.add_argument("--K",type=int,help="clusters number",default=30)
    parser.add_argument("--distance_mode",type=str,help="l1,l2,linf or cosine",default="cosine")
    parser.add_argument("--compress_method",type=str,help="pca, random or no_compress",default="pca")
    parser.add_argument("--init_method",type=str,help="random or farest",default="random")
    parser.add_argument("--compressed_dim",type=int,help="dim for compressed vectors if compress_method isn't no_compress",default=50)
    parser.add_argument("--seed",type=int,help="random seed",default=42)
    args = parser.parse_args()
    logger = loguru.logger
    random_seed=args.seed
    np.random.seed(random_seed)
    K = args.K
    logger_int = None
    distance_mode = args.distance_mode
    compress_method = args.compress_method
    init_method = args.init_method
    compressed_dim = args.compressed_dim
    save_dir = os.path.join("log",f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_{distance_mode}_{compress_method}_{init_method}_{compressed_dim}")
    os.makedirs(save_dir, exist_ok=True)
    if logger_int is not None:
        logger.remove(logger_int)
    logger_int = logger.add(os.path.join(save_dir,"logger.log"), rotation="10 MB", level="INFO", backtrace=True, diagnose=True, enqueue=True)
    logger.info(f"save to {save_dir}")
    images, labels = get_dataset(compressed_dim=compressed_dim,compress_method=compress_method)

    if args.mode == "estimate_k":
        elbow_k((images, labels),max_k=80)
    elif args.mode == "kmeans":
        clusters, centroids = kmeans((images,labels), K,max_iterations=200,eval_freq=3,distance_mode=distance_mode,save_dir=save_dir,init=init_method)
        accuracy = eval(clusters, labels, K)
        randome_num = 2000
        random_indices = np.random.choice(len(images), randome_num, replace=False)
        label_color_dict = generate_color_map(labels[random_indices])
        plot_tsne(images,centroids,os.path.join(save_dir,f"final_tsne_{K}.png"),clusters,random_indices,labels,label_color_dict)
        logger.info(f"Accuracy:{accuracy}")
    else:
        raise ValueError