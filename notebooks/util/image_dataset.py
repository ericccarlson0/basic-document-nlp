
# The base context manager for image datasets.
class ImageDataset(object):
    def __init__(self, directory: str):
        self._dir = directory

# A context manager.
# It represents a CSV image dataset resource.
class CsvImageDataset(ImageDataset):
    def __init__(self, directory: str):
        super(CsvImageDataset, self).__init__(directory)

    def __enter__(self):
        self._file = open(self._dir, 'r')
        return CsvIterator(self._file)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

# A context manager.
# It represents a TXT image dataset resource.
class TxtImageDataset(ImageDataset):
    def __init__(self, directory: str):
        super(TxtImageDataset, self).__init__(directory)

    def __enter__(self):
        self._file = open(self._dir, 'r')
        return TxtIterator(self._file)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()


# The base iterator for image datasets.
class ImageDataIterator(object):
    def __init__(self, file: iter):
        self._file = file

    def __iter__(self):
        return self

# An iterator with access to an open TXT resource.
# It returns pairs of (image directory, image label).
class TxtIterator(ImageDataIterator):
    def __init__(self, file: iter):
        super(TxtIterator, self).__init__(file)

    def __next__(self):
        line = next(self._file)
        img_dir, label = line.split()
        label = int(label)

        return img_dir, label

# An iterator with access to an open CSV resource.
# It returns pairs of (image directory, image label).
class CsvIterator(ImageDataIterator):
    def __init__(self, file: iter):
        super(CsvIterator, self).__init__(file)

    def __next__(self):
        line = next(self._file)
        img_dir, label = line.split(',')
        label = int(label)

        return img_dir, label