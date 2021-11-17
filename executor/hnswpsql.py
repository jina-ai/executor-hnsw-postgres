__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

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
    Production-ready, scalable Indexer for the Jina neural search framework.

    Combines the reliability of PostgreSQL with the speed and efficiency of the
    HNSWlib nearest neighbor library.
    """

    def __init__(
        self, total_shards: Optional[int] = None, startup_sync: bool = True, **kwargs
    ):
        """
        :param startup_sync: whether to sync from PSQL into HNSW on start-up
        :param total_shards: the total nr of shards that this shard is part of.
        :param limit: (HNSW) Number of results to get for each query document in
        search
        :param metric: (HNSW) Distance metric type. Can be 'euclidean',
        'inner_product', or 'cosine'
        :param dim: (HNSW) dimensionality of vectors to index
        :param max_elements: (HNSW) maximum number of elements (vectors) to index
        :param ef_construction: (HNSW) construction time/accuracy trade-off
        :param ef_query: (HNSW) query time accuracy/speed trade-off. High is more
        accurate but slower
        :param max_connection: (HNSW) The maximum number of outgoing connections in
        the graph (the "M" parameter)
        :param is_distance: (HNSW) if distance metric needs to be reinterpreted as
        similarity
        :param last_timestamp: (HNSW) the last time we synced into this HNSW index
        :param traversal_paths: (PSQL) default traversal paths on docs
        (used for indexing, delete and update), e.g. ['r'], ['c']
        :param hostname: (PSQL) hostname of the machine
        :param port: (PSQL) the port
        :param username: (PSQL) the username to authenticate
        :param password: (PSQL) the password to authenticate
        :param database: (PSQL) the database
        :param table: (PSQL) the table to use
        :param return_embeddings: (PSQL) whether to return embeddings on
        search
        :param dry_run: (PSQL) If True, no database connection will be built
        :param partitions: (PSQL) the number of shards to distribute
         the data (used when syncing into HNSW)

        NOTE:

        - `total_shards` is REQUIRED in k8s, since there
        `runtime_args.parallel` is always 1
        - some arguments are passed to the inner classes. They are documented
        here for easier reference
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

        self._kv_indexer: Optional[PostgreSQLStorage] = None
        self._vec_indexer: Optional[HnswlibSearcher] = None
        self._init_kwargs = kwargs

        (
            self._kv_indexer,
            self._vec_indexer,
        ) = self._init_executors(kwargs)
        if startup_sync:
            self._sync()

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

        if rebuild or self._vec_indexer is None:
            self._vec_indexer = HnswlibSearcher(**self._init_kwargs)

        iterator = self._kv_indexer._get_delta(
            shard_id=self.runtime_args.pea_id,
            total_shards=self.total_shards,
            timestamp=timestamp,
        )
        self._vec_indexer.sync(iterator)

    def _init_executors(
        self, _init_kwargs
    ) -> Tuple[PostgreSQLStorage, HnswlibSearcher]:
        kv_indexer = PostgreSQLStorage(dump_dtype=np.float32, **_init_kwargs)
        vec_indexer = HnswlibSearcher(**_init_kwargs)
        return kv_indexer, vec_indexer

    @requests(on='/index')
    def index(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """Index new documents

        NOTE: PSQL has a uniqueness constraint on ID

        :param docs: the Documents to index
        :param parameters: dictionary with options for indexing

        Keys accepted:

            - 'traversal_paths' (list): traversal path for the docs
        """
        self._kv_indexer.add(docs, parameters, **kwargs)

    @requests(on='/update')
    def update(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """Update existing documents

        :param docs: the Documents to update
        :param parameters: dictionary with options for updating

        Keys accepted:

            - 'traversal_paths' (list): traversal path for the docs
        """
        self._kv_indexer.update(docs, parameters, **kwargs)

    @requests(on='/delete')
    def delete(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """Update existing documents

        :param docs: the Documents to update
        :param parameters: dictionary with options for updating

        Keys accepted:

            - 'traversal_paths' (list): traversal path for the docs
            - 'soft_delete' (bool, default `True`): whether to perform soft delete
            (doc is marked as empty but still exists in db, for retrieval purposes)
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

        :return: DocumentArray with one Document with tags 'psql_docs', 'hnsw_docs',
        'last_sync', 'pea_id'
        """
        psql_docs = self._kv_indexer.size
        hnsw_docs = self._vec_indexer.size
        last_sync = self._vec_indexer.last_timestamp
        last_sync = last_sync.isoformat()
        status = {
            'psql_docs': psql_docs,
            'hnsw_docs': hnsw_docs,
            'last_sync': last_sync,
            'pea_id': self.runtime_args.pea_id,
        }
        return DocumentArray([Document(tags=status)])

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        """Search the vec embeddings in HNSW and then lookup the metadata in PSQL

        :param docs: `Document` with `.embedding` the same shape as the
            `Documents` stored in the `HNSW` index. The ids of the `Documents`
            stored in `HNSW` need to exist in the PSQL.
            Otherwise you will not get back the original metadata.
        :param parameters: dictionary for parameters for the search operation


            - 'traversal_paths' (list[str]): traversal path for the docs
            - 'limit' (int): nr of matches to get per Document
            - 'ef_query' (int): query time accuracy/speed trade-off. High is more
            accurate but slower

        :return: The `HNSWSearcher` attaches matches to the `Documents` sent as
            inputs with the id of the match, and its embedding.
            Then, the `PostgreSQLStorage` retrieves the full metadata
            (original text or image blob) and attaches
            those to the Document. You receive back the full Documents as matches
            to your search Documents.
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
