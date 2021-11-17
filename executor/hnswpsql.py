import copy
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np
from jina import Executor, requests, DocumentArray, Document
from jina.logging.logger import JinaLogger

from .hnswlib_searcher import HnswlibSearcher
from .postgres_indexer import PostgreSQLStorage


class HNSWPostgresIndexer(Executor):
    """
    TODO

    """

    def __init__(self,
                 total_shards: Optional[int] = None,
                 startup_sync: bool = True,
                 rebuild: bool = True,
                 **kwargs):
        """
        :param startup_sync: whether to sync from PSQL into HNSW on start-up
        :param total_shards: the total nr of shards that this shard is part of.
        :param rebuild: assume all sync operations should completely rebuild HNSW

            NOTE: `total_shards` is REQUIRED in k8s, since there
            `runtime_args.parallel` is always 1
        """
        super().__init__(**kwargs)
        self.logger = JinaLogger(getattr(self.metas, 'name', self.__class__.__name__))

        if total_shards is None:
            self.total_shards = getattr(self.runtime_args, 'parallel', None)
        else:
            self.total_shards = total_shards

        if self.total_shards is None:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )
        else:
            # shards is passed as str from Flow.add in yaml
            self.total_shards = int(self.total_shards)

        self.rebuild = rebuild
        self._kv_indexer: Optional[PostgreSQLStorage] = None
        self._vec_indexer: Optional[HnswlibSearcher] = None
        self._init_kwargs = kwargs

        (
            self._kv_indexer,
            self._vec_indexer,
        ) = self._init_executors(kwargs)
        if startup_sync:
            self._sync(self.rebuild)

    @requests(on='/sync')
    def sync(self, parameters: Dict, **kwargs):
        """
        Perform a sync between PSQL and HNSW

        :param parameters: dictionary with options for sync

        Keys accepted:
            - 'rebuild' (bool): whether to rebuild HNSW or do
            incremental syncing
            - 'timestamp' (str): ISO-formatted timestamp string. Time
            from which to get data for syncing into HNSW
        """
        self._sync(**parameters)

    def _sync(self, rebuild: bool = False, timestamp: str = None, **kwargs):
        if timestamp is None:
            if rebuild:
                timestamp = datetime.min
            elif self._vec_indexer.last_timestamp:
                timestamp = self._vec_indexer.last_timestamp
            else:
                self.logger.error(
                    f'No timestamp provided in parameters: '
                    f'and vec_indexer.last_timestamp'
                    f'was None. Cannot do sync'
                )
                return
        else:
            timestamp = datetime.fromisoformat(timestamp)

        if rebuild:
            self._vec_indexer = HnswlibSearcher(**self._init_kwargs)
        else:
            iterator = self._kv_indexer._get_delta(
                shard_id=self.runtime_args.pea_id,
                total_shards=self.total_shards,
                timestamp=timestamp,
            )
            self._vec_indexer.sync(iterator)

    def _init_executors(self, _init_kwargs) -> Tuple[PostgreSQLStorage,
                                                     HnswlibSearcher]:
        kv_indexer = PostgreSQLStorage(dump_dtype=np.float32, **_init_kwargs)
        vec_indexer = HnswlibSearcher(**_init_kwargs)
        return kv_indexer, vec_indexer

    @requests(on='/index')
    def index(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """Index new documents

        NOTE: PSQL has a uniqueness constraint on ID
        """
        self._kv_indexer.add(docs, parameters, **kwargs)

    @requests(on='/update')
    def update(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Update documents in PSQL, based on id
        """
        self._kv_indexer.update(docs, parameters, **kwargs)

    @requests(on='/delete')
    def delete(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Delete docs from PSQL, based on id.

        By default, it will be a soft delete, where the entry is left in the DB,
        but its data will be set to None
        """
        if 'soft_delete' not in parameters:
            parameters['soft_delete'] = True

        self._kv_indexer.delete(docs, parameters, **kwargs)

    @requests(on='/clear')
    def clear(self, **kwargs):
        """
        Delete data from PSQL and HNSW

        """
        self._kv_indexer.clear()
        self._vec_indexer = HnswlibSearcher(**self._init_kwargs)

    @requests(on='/status')
    def status(self, **kwargs):
        """
        Get information on status of this Indexer inside a Document's tags
        """
        psql_docs = self._kv_indexer.size
        hnsw_docs = self._vec_indexer.size
        last_sync = self._vec_indexer.last_timestamp
        last_sync = last_sync.isoformat()
        status = {
            'psql_docs': psql_docs,
            'hnsw_docs': hnsw_docs,
            'last_sync': last_sync,
            'pea_id': self.runtime_args.pea_id
        }
        return DocumentArray([
            Document(tags=status)
        ])

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        """
        Search the vec embeddings in Faiss and then lookup the metadata in PSQL

        :param docs: `Document` with `.embedding` the same shape as the
            `Documents` stored in the `FaissSearcher`. The ids of the `Documents`
            stored in `FaissSearcher` need to exist in the `PostgreSQLStorage`.
            Otherwise you will not get back the original metadata.
        :param parameters: dictionary to define the ``traversal_paths``. This will
            override the default parameters set at init.

        :return: The `FaissSearcher` attaches matches to the `Documents` sent as
        inputs,
            with the id of the match, and its embedding. Then, the `PostgreSQLStorage`
            retrieves the full metadata (original text or image blob) and attaches
            those to the Document. You receive back the full Document.

        """
        if self._kv_indexer and self._vec_indexer:
            self._vec_indexer.search(docs, parameters)
            kv_parameters = copy.deepcopy(parameters)
            kv_parameters['traversal_paths'] = [
                path + 'm' for path in kv_parameters.get('traversal_paths', ['r'])
            ]
            self._kv_indexer.search(docs, kv_parameters)
        else:
            self.logger.warning('Indexers have not been initialized. Empty results')
            return
