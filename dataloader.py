from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from scipy.sparse import issparse

class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class Yale(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'Yale.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Yale.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Yale.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'Yale.mat')['Y'].transpose()

        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class HW(Dataset):
    def __init__(self, path):
        # 加载六个视图的数据
        data1 = scipy.io.loadmat(path + 'HW.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'HW.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'HW.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'HW.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'HW.mat')['X5'].astype(np.float32)
        data6 = scipy.io.loadmat(path + 'HW.mat')['X6'].astype(np.float32)

        labels = scipy.io.loadmat(path + 'HW.mat')['Y'].transpose()

        # 存储视图数据和标签
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        # 返回数据集的大小
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回六个视图的数据及标签
        return [
            torch.from_numpy(self.x1[idx]),
            torch.from_numpy(self.x2[idx]),
            torch.from_numpy(self.x3[idx]),
            torch.from_numpy(self.x4[idx]),
            torch.from_numpy(self.x5[idx]),
            torch.from_numpy(self.x6[idx])
        ], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class ORL(Dataset):
    def __init__(self, path):
        # 加载 ORL 数据集的三个视图数据
        data1 = scipy.io.loadmat(path + 'ORL.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'ORL.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'ORL.mat')['X3'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'ORL.mat')['Y'].transpose()

        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括三个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Citeseer(Dataset):
    def __init__(self, path):
        # 加载 Citeseer 数据集的两个视图数据
        data1 = scipy.io.loadmat(path + 'Citeseer.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Citeseer.mat')['X2'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'Citeseer.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括两个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], \
               torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        # 加载 CCV 数据集的 3 个视图数据
        data1 = scipy.io.loadmat(path + 'CCV.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'CCV.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'CCV.mat')['X3'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'CCV.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 3 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class cifar_10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class cifar_100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)],self.Y[idx], torch.from_numpy(np.array(idx)).long()

class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Yale(Dataset):
    def __init__(self, path):
        # 加载 Yale 数据集的 3 个视图数据
        data1 = scipy.io.loadmat(path + 'Yale.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Yale.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Yale.mat')['X3'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'Yale.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 3 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Mfeat(Dataset):
    def __init__(self, path):
        # 加载 Mfeat 数据集的 6 个视图数据
        data1 = scipy.io.loadmat(path + 'Mfeat.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Mfeat.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Mfeat.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'Mfeat.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'Mfeat.mat')['X5'].astype(np.float32)
        data6 = scipy.io.loadmat(path + 'Mfeat.mat')['X6'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'Mfeat.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 6 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx]),
                torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class Hdigit(Dataset):
    def __init__(self, path):
        # 加载 Hdigit 数据集的 2 个视图数据
        data1 = scipy.io.loadmat(path + 'Hdigit.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Hdigit.mat')['X2'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'Hdigit.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 2 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class HW2sources(Dataset):
    def __init__(self, path):
        # 加载 Hdigit 数据集的 2 个视图数据
        data1 = scipy.io.loadmat(path + 'HW2sources.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'HW2sources.mat')['X2'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'HW2sources.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 2 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Leaves100(Dataset):
    def __init__(self, path):
        # 加载 100leaves 数据集的 3 个视图数据
        data1 = scipy.io.loadmat(path + '100Leaves.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + '100Leaves.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + '100Leaves.mat')['X3'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + '100Leaves.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 3 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class NGs(Dataset):
    def __init__(self, path):
        # 加载 NGs 数据集的 3 个视图数据
        data1 = scipy.io.loadmat(path + 'NGs.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'NGs.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'NGs.mat')['X3'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'NGs.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 3 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class ALOL100(Dataset):
    def __init__(self, path):
        # 加载 ALOL-100 数据集的多个视图数据
        data1 = scipy.io.loadmat(path + 'ALOL-100.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'ALOL-100.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'ALOL-100.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'ALOL-100.mat')['X4'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'ALOL-100.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 four 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx])], \
            torch.from_numpy(self.y[idx]), torch.tensor(idx, dtype=torch.long)

class Cora(Dataset):
    def __init__(self, path):
        # 加载 Cora 数据集的多个视图数据
        data1 = scipy.io.loadmat(path + 'Cora.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Cora.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Cora.mat')['X3'].astype(np.float32)
        # 加载标签数据
        labels = scipy.io.loadmat(path + 'Cora.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 3 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Caltech101_7(Dataset):
    def __init__(self, path):
        # 加载 Caltech101-7d 数据集的多个视图数据
        data1 = scipy.io.loadmat(path + 'Caltech101-7.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Caltech101-7.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Caltech101-7.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'Caltech101-7.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'Caltech101-7.mat')['X5'].astype(np.float32)
        data6 = scipy.io.loadmat(path + 'Caltech101-7.mat')['X6'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'Caltech101-7.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 6 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx])], \
            torch.from_numpy(self.y[idx]), torch.tensor(idx, dtype=torch.long)


class Caltech101_20(Dataset):
    def __init__(self, path):
        # 加载 Caltech101-20 数据集的多个视图数据
        data1 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X5'].astype(np.float32)
        data6 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X6'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'Caltech101-20.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.x6 = data6
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 6 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx]),
                torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx])], \
            torch.from_numpy(self.y[idx]), torch.tensor(idx, dtype=torch.long)


class MSRC_v1(Dataset):
    def __init__(self, path):
        # 加载 MSRC-v1 数据集的多个视图数据
        data1 = scipy.io.loadmat(path + 'MSRC-v1.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'MSRC-v1.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'MSRC-v1.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'MSRC-v1.mat')['X4'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'MSRC-v1.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 four 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx])], \
            torch.from_numpy(self.y[idx]), torch.tensor(idx, dtype=torch.long)

class WikipediaArticles(Dataset):
    def __init__(self, path):
        # 加载 WikipediaArticles 数据集的两个视图数据
        data1 = scipy.io.loadmat(path + 'WikipediaArticles.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'WikipediaArticles.mat')['X2'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'WikipediaArticles.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.view1 = data1
        self.view2 = data2
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.view1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 2 个视图的数据和对应的标签）
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], \
               torch.from_numpy(self.y[idx]), torch.tensor(idx, dtype=torch.long)

class BBCSport(Dataset):
    def __init__(self, path):
        """
        初始化 BBCSport 数据集。

        参数:
            path (str): 数据集文件的路径。
        """
        # 加载 BBCSport 数据集的两个视图数据
        data = scipy.io.loadmat(path + 'BBCSport.mat')
        self.view1 = data['X1'].astype(np.float32)  # 视图 1
        self.view2 = data['X2'].astype(np.float32)  # 视图 2

        # 如果数据是稀疏矩阵，转换为密集矩阵
        if issparse(self.view1):
            self.view1 = self.view1.toarray()
        if issparse(self.view2):
            self.view2 = self.view2.toarray()

        # 加载标签数据并转置以匹配样本维度
        labels = data['Y'].transpose()

        # 将数据存储为类的属性
        self.y = labels

    def __len__(self):
        """
        返回数据集的样本数。
        """
        return self.view1.shape[0]

    def __getitem__(self, idx):
        """
        返回指定索引的样本（包括 2 个视图的数据和对应的标签）。

        参数:
            idx (int): 样本索引。

        返回:
            tuple: 包含两个视图的数据、标签和索引。
        """
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], \
               torch.from_numpy(self.y[idx]), torch.tensor(idx, dtype=torch.long)
class ThreeSources(Dataset):
    def __init__(self, path):
        # 加载 3Sources 数据集的三个视图数据
        data1 = scipy.io.loadmat(path + '3Sources.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + '3Sources.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + '3Sources.mat')['X3'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + '3Sources.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.view1 = data1
        self.view2 = data2
        self.view3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.view1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 3 个视图的数据和对应的标签）
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(self.view3[idx])], \
               torch.from_numpy(self.y[idx]), torch.tensor(idx, dtype=torch.long)
class WebKB(Dataset):
    def __init__(self, path):
        # 加载 WebKB 数据集的三个视图数据
        data1 = scipy.io.loadmat(path + 'WebKB.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'WebKB.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'WebKB.mat')['X3'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'WebKB.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括三个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]),
                torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class ThreeRing(Dataset):
    def __init__(self, path):
        # 加载 Three Ring 数据集的 2 个视图数据
        data1 = scipy.io.loadmat(path + 'ThreeRing.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'ThreeRing.mat')['X2'].astype(np.float32)

        # 加载标签数据
        labels = scipy.io.loadmat(path + 'ThreeRing.mat')['Y'].transpose()

        # 将数据存储为类的属性
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        # 返回数据集的样本数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        # 返回指定索引的样本（包括 2 个视图的数据和对应的标签）
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], \
            torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "Cora":
        dataset = Cora('./data/')
        dims = [2708,1433,2706]
        view = 3
        data_size =2708
        class_num = 7
    elif dataset == "ThreeRing":
        dataset = ThreeRing('./data/')
        dims = [2, 2]  # 3 个视图的维度分别为4096, 3304, 6750
        view = 2 # 视图数为2
        data_size = 300  # 数据大小为165
        class_num = 3 # 假设类别数为15（根据数据集实际类别数调整）
    elif dataset == "Citeseer":
        dataset = Citeseer('./data/')  # 加载 Citeseer 数据集
        dims = [3312, 3703]  # 2 个视图的维度分别为 3312, 3703
        view = 2  # 视图数为 2
        data_size = 3312  # 数据大小为 3312
        class_num = 6  # 类别数为 6
    elif dataset == "WebKB":
        dataset = WebKB('./data/')  # 加载 WebKB 数据集
        dims = [1703, 230, 230]  # 3 个视图的维度分别为 1703, 230, 230
        view = 3  # 视图数为 3
        data_size = 203  # 数据大小为 203
        class_num = 3  # 类别数为 3
    elif dataset == "3Sources":
        dataset = ThreeSources('./data/')  # 加载 3Sources 数据集
        dims = [3560, 3631, 3068]  # 3 个视图的维度分别为 3560, 3631, 3068
        view = 3  # 视图数为 3
        data_size = 169  # 数据大小为 169
        class_num = 6  # 类别数为 6
    elif dataset == "HW2sources":
        dataset = HW2sources('./data/')  # 加载 3Sources 数据集
        dims = [784,256]  # 3 个视图的维度分别为 3560, 3631, 3068
        view = 2  # 视图数为 3
        data_size = 2000  # 数据大小为 169
        class_num = 10  # 类别数为 6
    elif dataset == "BBCSport":
        dataset = BBCSport('./data/')  # 加载 BBCSport 数据集
        dims = [3183, 3203]  # 2 个视图的维度分别为 3183, 3203
        view = 2  # 视图数为 2
        data_size = 544  # 数据大小为 544
        class_num = 5  # 类别数为 5
    elif dataset == "WikipediaArticles":
        dataset = ThreeRing('./data/')
        dims = [128, 10]
        view = 2 # 视图数为2
        data_size = 693  # 数据大小为165
        class_num = 10# 假设类别数为15（根据数据集实际类别数调整）
    elif dataset == "MSRC-v1":
        dataset = MSRC_v1('./data/')
        dims = [24, 512, 256, 254]  # four 个视图的维度
        view = 4  # 视图数为 four
        data_size = 210  # 数据大小为 210
        class_num = 7  # 假设类别数为 7
    elif dataset == "ALOL100":
        dataset = ALOL100('./data/')
        dims = [77, 13, 64, 125]  # four 个视图的维度
        view = 4  # 视图数为 four
        data_size = 10800  # 数据大小为 10800
        class_num = 100  # 假设类别数为 100（根据数据集实际类别数调整）

    elif dataset == "Caltech101-7":
        dataset = Caltech101_7('./data/')
        dims = [48, 40, 254, 1984, 512, 928]  # 6 个视图的维度
        view = 6  # 视图数为 6
        data_size = 1474  # 数据大小为 1474
        class_num = 7  # 假设类别数为 7（根据数据集实际类别数调整）

    elif dataset == "Caltech101-20":
        dataset = Caltech101_7('./data/')
        dims = [48, 40, 254, 1984, 512, 928]  # 6 个视图的维度
        view = 6  # 视图数为 6
        data_size = 2386  # 数据大小为 1474
        class_num = 20  # 假设类别数为 7（根据数据集实际类别数调整）

    elif dataset == "Yale":
        dataset = Yale('./data/')
        dims = [4096, 3304, 6750]  # 3 个视图的维度分别为4096, 3304, 6750
        view = 3  # 视图数为3
        data_size = 165  # 数据大小为165
        class_num = 15  # 假设类别数为15（根据数据集实际类别数调整）

    elif dataset == "NGs":
        dataset = NGs('./data/')
        dims = [2000, 2000, 2000]  # 3 个视图的维度均为500
        view = 3  # 视图数为3
        data_size = 500 # 数据大小为2000
        class_num = 5  # 类别数为5

    elif dataset == "100leaves":
        dataset = Leaves100('./data/')
        dims = [64, 64, 64]  # 3 个视图的维度均为64
        view = 3  # 视图数为3
        data_size = 1600  # 数据大小为1600
        class_num = 100  # 类别数为100

    elif dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]  # 2 个视图的维度分别为784, 256
        view = 2  # 视图数为2
        data_size = 10000  # 数据大小为10000
        class_num = 10  # 类别数为2

    elif dataset == "Mfeat":
        dataset = Mfeat('./data/')
        dims = [216, 76, 64, 6, 240, 47]  # 6 个视图的维度分别为216, 76, 64, 6, 240, 47
        view = 6  # 视图数为6
        data_size = 2000  # 数据大小为2000
        class_num = 10  # 类别数为10（根据数据集的实际类别数调整）

    elif dataset == "HW":
        dataset = HW('./data/')
        dims = [216, 76, 64, 6, 240, 47]  # 维度分别为216, 76, 64, 6, 240, 47
        view = 6  # 视图数为6
        data_size = 2000  # 数据大小为2000
        class_num = 10  # 类别数为10
    elif dataset == "Yale":
        dataset = Yale('./data/')
        dims = [4096, 3304, 6750]  # 维度分别为4096, 3304, 6750
        view = 3  # 视图数为3
        data_size = 165  # 数据大小为165
        class_num = 15  # 类别数为15
    elif dataset == "ORL":
        dataset = ORL('./data/')
        dims = [4096, 3304, 6750]  # 假设三个视图的维度分别为1024, 800, 1600
        view = 3  # 视图数为3
        data_size = 400  # 假设数据集中有400个样本
        class_num = 40  # 类别数为40

    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Synthetic3d":
        dataset = synthetic3d('./data/')
        dims = [3,3,3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Prokaryotic":
        dataset = prokaryotic('./data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "Prokaryotic":
        dataset = prokaryotic('./data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "Cifar10":
        dataset = cifar_10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "Cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [20, 20, 20]  # 3 个视图的维度分别为20, 20, 20
        view = 3  # 视图数为3
        data_size = 6773  # 数据大小为6773
        class_num = 10  # 类别数为10（根据数据集的实际类别数调整）

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
