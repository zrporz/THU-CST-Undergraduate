import numpy as np

def page_rank(adj_matrix_, beta=0.85, max_iter=1000, tol=1e-3):
    adj_matrix = adj_matrix_.copy()
    n = len(adj_matrix)
    d = np.sum(adj_matrix, axis=0)


    # 初始化页面排名向量
    pr = np.ones(n)
    for _ in range(max_iter):
        new_pr = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if d[j] != 0:
                    new_pr[i] += beta * adj_matrix[i][j]/d[j] * pr[j]
                else:
                    new_pr[i] += beta *1/n * pr[j]
            new_pr[i] += (1-beta) * 1/n
        
        # 收敛判断
        if np.linalg.norm(new_pr - pr,ord=np.inf) < tol:
            return new_pr
        
        pr = new_pr
                
    return pr

    
# 示例用法
adj_matrix = np.array([
    [0., 0, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 1, 0]])

pr = page_rank(adj_matrix)
print(pr)
