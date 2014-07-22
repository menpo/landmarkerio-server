import os.path as p
from pathlib import Path
from flask import safe_join
import abc

from landmarkerio import COLLECTION_DIRNAME, COLLECTION_EXT


def load_collection(path):
    with open(path, 'rb') as f:
        collection = [l.strip() for l in f.readlines()]
    return [l for l in collection if len(l) > 0]


class CollectionAdapter(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def collection_ids(self):
        pass

    @abc.abstractmethod
    def collection_list(self, collection_id):
        pass


class FileCollectionAdapter(CollectionAdapter):

    def __init__(self, collection_dir=None):
        if collection_dir is None:
            # try the user folder
            user_collections = p.expanduser(p.join('~', COLLECTION_DIRNAME))
            if p.isdir(user_collections):
                collection_dir = user_collections
            else:
                raise ValueError("No collection dir provided and "
                                 "{} doesn't exist".format(user_collections))
        self.collection_dir = Path(p.abspath(p.expanduser(collection_dir)))
        print ('collections: {}'.format(self.collection_dir))

    def collection_ids(self):
        collection_paths = self.collection_dir.glob('*' + COLLECTION_EXT)
        return [c.stem for c in collection_paths]

    def collection_list(self, collection_id):
        fp = safe_join(str(self.collection_dir), collection_id +
                                                 COLLECTION_EXT)
        return load_collection(fp)
