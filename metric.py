from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_true = y_true.copy()
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def valid(model, device, dataset, view, data_size, class_num, eval_h=False,
          epoch=None, batch_size=512, n_init=100, random_state=10):
    was_training = model.training
    model.eval()
    collected, labels, zparts, rparts = [], [], [[] for _ in range(view)], [[] for _ in range(view)]
    try:
        for xs, y, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            xs = [x.to(device) for x in xs]
            with torch.no_grad():
                _, zs, rs, H, _ = model(xs)
            collected.append(H.cpu().numpy())
            labels.append(torch.as_tensor(y).cpu().numpy().reshape(-1))
            if eval_h:
                for v in range(view):
                    zparts[v].append(zs[v].cpu().numpy())
                    rparts[v].append(rs[v].cpu().numpy())
        labels = np.concatenate(labels)
        if eval_h:
            for kind, parts in [('latent', zparts), ('local', rparts)]:
                for v, batches in enumerate(parts):
                    pred = KMeans(n_clusters=class_num, n_init=n_init,
                                  random_state=random_state).fit_predict(np.concatenate(batches))
                    nmi, ari, acc, pur = evaluate(labels, pred)
                    print(f'{kind} view {v}: ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f}')
        pred = KMeans(n_clusters=class_num, n_init=n_init,
                      random_state=random_state).fit_predict(np.concatenate(collected))
        nmi, ari, acc, pur = evaluate(labels, pred)
        print(f'Epoch {epoch}: ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f} PUR={pur:.4f}')
        return acc, nmi, pur
    finally:
        model.train(was_training)
