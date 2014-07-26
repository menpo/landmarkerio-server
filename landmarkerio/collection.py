from os.path import abspath, expanduser
from pathlib import Path
import abc

from landmarkerio import ALL_COLLECTION_ID, dirs_in_dir, FileExt


def load_collection(path):
    with open(str(path), 'rb') as f:
        collection = [l.strip() for l in f.readlines()]
    return [l for l in collection if len(l) > 0]


class CollectionAdapter(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def collection_ids(self):
        pass

    @abc.abstractmethod
    def collection(self, collection_id):
        pass

    def __str__(self):
        a = ('Serving {} collection(s):'.format(len(self.collection_ids())))
        b = '\n'.join(' - {} ({} assets)'.format(c, len(self.collection(c)))
                      for c in self.collection_ids())
        return a + '\n' + b


class FileCollectionAdapter(CollectionAdapter):

    def __init__(self, collection_dir):
        self.collection_dir = Path(abspath(expanduser(collection_dir)))
        print ('collections: {}'.format(self.collection_dir))
        collection_paths = self.collection_dir.glob('*' + FileExt.collection)
        self._collection = {c.stem: load_collection(c)
                            for c in collection_paths}

    def collection_ids(self):
        return self._collection.keys()

    def collection(self, collection_id):
        return self._collection[collection_id]


class AllCacheCollectionAdapter(CollectionAdapter):

    def __init__(self, cache_dir):
        cache_dir = Path(abspath(expanduser(cache_dir)))
        self._collection = [p.name for p in dirs_in_dir(cache_dir)]
        self._collection_ids = [ALL_COLLECTION_ID]

    def collection_ids(self):
        return self._collection_ids

    def collection(self, collection_id):
        if collection_id == ALL_COLLECTION_ID:
            return self._collection
        else:
            raise ValueError("Only valid collection_id "
                             "is '{}'".format(ALL_COLLECTION_ID))
