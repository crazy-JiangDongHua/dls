import numpy as np
from numpy.core.memmap import dtype
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img=np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        result=np.zeros_like(img)
        H, W, _=img.shape
        # 感觉x，y的定义是不是反了，很奇怪，相当于从（shift_x, shift_y）开始框一个img这么大的图，空的用zero填补
        if abs(shift_x) >= H or abs(shift_y) >= W:
            return result
        st_1, ed_1 = max(0, -shift_x), min(H - shift_x, H)
        st_2, ed_2 = max(0, -shift_y), min(W - shift_y, W)
        img_st_1, img_ed_1 = max(0, shift_x), min(H + shift_x, H)
        img_st_2, img_ed_2 = max(0, shift_y), min(W + shift_y, W)
        result[st_1:ed_1, st_2:ed_2, :] = img[img_st_1:img_ed_1, img_st_2:img_ed_2, :]
        return result
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                            range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.i=0
        if self.shuffle:
            indices=np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, 
                            range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.i==len(self.ordering):
            raise StopIteration
        # 可能只有x，也有可能是(x,y)，还有可能是(a,b,c,d)
        # 用np.array([])做index，即使len==1，索引后也不会降维
        # np.array([[1,2],[2,3]])[np.array([1])]=np.array([[2,3]])
        data=[Tensor(x) for x in self.dataset[self.ordering[self.i]]]
        self.i+=1
        return tuple(data)
        ### END YOUR SOLUTION

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    import gzip
    import struct
    with gzip.open(image_filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>4I', f.read(16))
        image = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    with gzip.open(label_filename, 'rb') as f:
        magic, num = struct.unpack('>2I', f.read(8))
        label = np.frombuffer(f.read(), dtype=np.uint8)
    image = image.astype(np.float32) / 255.0
    return image, label
    ### END YOUR CODE

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.image, self.label=parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X, y=self.image[index], self.label[index]
        # 如果index是list, tuple, 或是一维的ndarray
        # X.shape=(len(index), 28*28), y.shape=(len(index),)
        # 如果index是int
        # X.shape=(28*28,), y.shape=(len(index),)
        if self.transforms is not None:
            X=X.T.reshape((28, 28, -1))
            X=self.apply_transforms(X)
            X=X.transpose((2,0,1)).reshape((-1, 28*28))
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.image.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
