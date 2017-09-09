import numpy as np


def weights_cluster(weights, bits=5, epsilon=1e-11, verbose=False):
    orig_shape = weights.shape
    weights = weights.reshape([-1, 1])
    cluster_num = np.power(2, bits)
    centroids = np.linspace(start=np.min(weights),
                            stop=np.max(weights),
                            num=cluster_num).reshape([-1, 1])

    cnt = 0
    while True:
        dis = np.square(weights - centroids.T)
        belongs = np.argmin(dis, axis=1)

        old_c = np.copy(centroids)
        for j in range(centroids.shape[0]):
            if np.sum(belongs == j) > 0:
                centroids[j] = np.mean(weights[belongs == j, :])
        # loss = 0.0
        # for i in range(weights.shape[0]):
        #     loss += dis[i,belongs[i]]
        cnt += 1
        if np.sum(np.abs(old_c-centroids)) < epsilon:
            break

    if verbose:
        print "iter for %d times" % cnt
    return belongs.reshape(*orig_shape), centroids
