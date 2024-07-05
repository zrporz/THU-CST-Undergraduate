import numpy as np
import matplotlib.pyplot as plt
def plot_pca(vectors,output_path,label_list):
    from sklearn.decomposition import PCA

    # 使用PCA进行降维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)

    # 绘制PCA降维结果，并调整标签位置
    plt.figure(figsize=(20, 16))
    plt.scatter(pca_result[:, 0], pca_result[:, 1],color=label_list,linewidths=10)
    for i, word in enumerate(words):
        plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]), xytext=(5, 2), textcoords='offset points', fontsize=10)  # 设置标签的偏移量
    plt.title('PCA Word Embedding Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(output_path)

def plot_tsne(vectors,output_path,label_list):
    from sklearn.manifold import TSNE

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(vectors)

    # 绘制t-SNE降维结果
    plt.figure(figsize=(20, 16))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1],color=label_list,linewidths=10)
    for i, word in enumerate(words):
        plt.annotate(word, (tsne_result[i, 0], tsne_result[i, 1]),xytext=(5, 2), textcoords='offset points',  fontsize=10)
    plt.title('t-SNE Word Embedding Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(output_path)


word_vectors = {}
with open('100_word_vector.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        word = parts[0]
        vector = np.array([float(x) for x in parts[1].split()])
        word_vectors[word] = vector

# 提取单词和向量
words = list(word_vectors.keys())
vectors = np.array(list(word_vectors.values()))
color_map= plt.cm.rainbow(np.linspace(0, 1, 5))
label_list = [color_map[i//20] for i in range(100)]
plot_pca(vectors,"Visualization_PCA.png",label_list)
plot_tsne(vectors,"Visualization_TSNE.png",label_list)



