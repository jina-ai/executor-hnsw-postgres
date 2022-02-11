__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Generator, Tuple, List

import hnswlib
import numpy as np
from bidict import bidict
from jina import DocumentArray, Document
from jina.logging.logger import JinaLogger

GENERATOR_DELTA = Generator[
    Tuple[str, Optional[np.ndarray], Optional[datetime]], None, None
]

HNSW_TYPE = np.float32
DEFAULT_METRIC = 'cosine'


class HnswlibSearcher:
    """Hnswlib powered vector indexer.

    This indexer uses the HNSW algorithm to index and search for vectors. It does not
    require training, and can be built up incrementally.
    """

    def __init__(
        self,
        limit: int = 10,
        metric: str = DEFAULT_METRIC,
        dim: int = 0,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        ef_query: int = 50,
        max_connection: int = 16,
        dump_path: Optional[str] = None,
        traversal_paths: str = '@r',
        is_distance: bool = True,
        last_timestamp: datetime = datetime.fromtimestamp(0, timezone.utc),
        num_threads: int = -1,
        *args,
        **kwargs,
    ):
        """
        :param limit: Number of results to get for each query document in search
        :param metric: Distance metric type, can be 'euclidean', 'inner_product',
        or 'cosine'
        :param dim: The dimensionality of vectors to index
        :param max_elements: Maximum number of elements (vectors) to index
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_query: The query time accuracy/speed trade-off. High is more
            accurate but slower
        :param max_connection: The maximum number of outgoing connections in the
            graph (the "M" parameter)
        :param dump_path: The path to the directory from where to load, and where to
            save the index state
        :param traversal_paths: The default traversal path on docs (used for
        indexing, search and update), e.g. '@r', '@c', '@r,c'
        :param is_distance: Boolean flag that describes if distance metric need to
        be reinterpreted as similarities.
        :param last_timestamp: the last time we synced into this HNSW index
        :param num_threads: nr of threads to use during indexing. -1 is default
        """
        self.limit = limit
        self.metric = metric
        self.dim = dim
        self.max_elements = max_elements
        self.traversal_paths = traversal_paths
        self.ef_construction = ef_construction
        self.ef_query = ef_query
        self.max_connection = max_connection
        self.dump_path = dump_path
        self.is_distance = is_distance
        self.last_timestamp = last_timestamp
        self.num_threads = num_threads

        self.logger = JinaLogger(self.__class__.__name__)
        self._index = hnswlib.Index(space=self.metric_type, dim=self.dim)

        # TODO(Cristian): decide whether to keep this for eventual dump loading
        dump_path = self.dump_path or kwargs.get('runtime_args', {}).get(
            'dump_path', None
        )
        if dump_path is not None:
            self.logger.info('Starting to build HnswlibSearcher from dump data')

            self._index.load_index(
                f'{self.dump_path}/index.bin', max_elements=self.max_elements
            )
            with open(f'{self.dump_path}/ids.json', 'r') as f:
                self._ids_to_inds = bidict(json.load(f))

        else:
            self._init_empty_index()

        self._index.set_ef(self.ef_query)

    def _init_empty_index(self):
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.max_connection,
        )
        self._ids_to_inds = bidict()

    def search(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Attach matches to the Documents in `docs`, each match containing only the
        `id` of the matched document and the `score`.

        :param docs: An array of `Documents` that should have the `embedding` property
            of the same dimension as vectors in the index
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. Supported keys are
            `traversal_paths`, `limit` and `ef_query`.
        """
        if docs is None:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        docs_search = docs[traversal_paths]
        if len(docs_search) == 0:
            return

        ef_query = parameters.get('ef_query', self.ef_query)
        limit = int(parameters.get('limit', self.limit))

        self._index.set_ef(ef_query)

        if limit > len(self._ids_to_inds):
            limit = len(self._ids_to_inds)

        embeddings_search = docs_search.embeddings
        if embeddings_search.shape[1] != self.dim:
            raise ValueError(
                'Query documents have embeddings with dimension'
                f' {embeddings_search.shape[1]}, which does not match the dimension '
                f'of'
                f' the index ({self.dim})'
            )

        indices, dists = self._index.knn_query(docs_search.embeddings, k=limit)

        for i, (indices_i, dists_i) in enumerate(zip(indices, dists)):
            for idx, dist in zip(indices_i, dists_i):
                match = Document(id=self._ids_to_inds.inverse[idx])
                if self.is_distance:
                    match.scores[self.metric] = dist
                elif self.metric in ["inner_product", "cosine"]:
                    match.scores[self.metric] = 1 - dist
                elif self.metric == 'euclidean':
                    match.scores[self.metric] = 1 / (1 + dist)
                else:
                    match.scores[self.metric] = dist
                docs_search[i].matches.append(match)

    def index(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Index the Documents' embeddings. If the document is already in index, it
        will be updated.

        :param docs: Documents whose `embedding` to index.
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `traversal_paths`.
        """
        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        if docs is None:
            return

        docs_to_index = docs[traversal_paths]
        if len(docs_to_index) == 0:
            return

        embeddings = docs_to_index.embeddings
        if embeddings.shape[-1] != self.dim:
            raise ValueError(
                f'Attempted to index vectors with dimension'
                f' {embeddings.shape[-1]}, but dimension of index is {self.dim}'
            )

        ids = docs_to_index[:,'id']
        index_size = self._index.element_count
        docs_inds = []
        for id in ids:
            if id not in self._ids_to_inds:
                docs_inds.append(index_size)
                index_size += 1
            else:
                self.logger.info(f'Document with id {id} already in index, updating.')
                docs_inds.append(self._ids_to_inds[id])
        self._add(embeddings, ids, docs_inds)

    def _add(self, embeddings, ids, docs_inds: Optional[List[int]] = None):
        if docs_inds is None:
            docs_inds = list(
                range(self._index.element_count, self._index.element_count + len(ids))
            )
        self._index.add_items(embeddings, ids=docs_inds, num_threads=self.num_threads)
        self._ids_to_inds.update({_id: ind for _id, ind in zip(ids, docs_inds)})

    def update(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Update the Documents' embeddings. If a Document is not already present in
        the index, it will get ignored, and a warning will be raised.

        :param docs: Documents whose `embedding` to update.
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `traversal_paths`.
        """
        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        if docs is None:
            return

        docs_to_update = docs[traversal_paths]
        if len(docs_to_update) == 0:
            return

        # TODO(Cristian): don't recreate DA if the ids all exist.
        # we are punishing everyone the same
        # instead of rewarding people that send updated with existing ids
        doc_inds, docs_filtered = [], []
        for doc in docs_to_update:
            if doc.id not in self._ids_to_inds:
                self.logger.warning(
                    f'Attempting to update document with id {doc.id} which is not'
                    ' indexed, skipping. To add documents to index, use the /index'
                    ' endpoint'
                )
            else:
                docs_filtered.append(doc)
                doc_inds.append(self._ids_to_inds[doc.id])
        docs_filtered = DocumentArray(docs_filtered)

        embeddings = docs_filtered.embeddings
        if embeddings.shape[-1] != self.dim:
            raise ValueError(
                f'Attempted to update vectors with dimension'
                f' {embeddings.shape[-1]}, but dimension of index is {self.dim}'
            )

        self._index.add_items(embeddings, ids=doc_inds, num_threads=self.num_threads)

    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters to the request. Should contain the list of ids
            of entries (Documents) to delete under the `ids` key
        """
        deleted_ids = parameters.get('ids', [])

        for _id in set(deleted_ids).intersection(self._ids_to_inds.keys()):
            ind = self._ids_to_inds[_id]
            self._index.mark_deleted(ind)
            del self._ids_to_inds[_id]

    def dump(self, parameters: Dict = {}, **kwargs):
        """Save the index and document ids.

        The index and ids will be saved separately for each shard.

        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `dump_path`.
        """

        dump_path = parameters.get('dump_path', self.dump_path)
        if dump_path is None:
            raise ValueError(
                'The `dump_path` must be provided to save the indexer state.'
            )

        self._index.save_index(f'{dump_path}/index.bin')
        with open(f'{dump_path}/ids.json', 'w') as f:
            json.dump(dict(self._ids_to_inds), f)

    def clear(self, **kwargs):
        """Clear the index of all entries."""
        self._index = hnswlib.Index(space=self.metric_type, dim=self.dim)
        self._init_empty_index()
        self._index.set_ef(self.ef_query)

    def status(self) -> Dict:
        """Return the status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = {
            'count_deleted': self._index.element_count - len(self._ids_to_inds),
            'count_indexed': self._index.element_count,
            'count_active': self.size,
        }
        return status

    @property
    def size(self):
        return len(self._ids_to_inds)

    @property
    def metric_type(self):
        if self.metric == 'euclidean':
            metric_type = 'l2'
        elif self.metric == 'cosine':
            metric_type = 'cosine'
        elif self.metric == 'inner_product':
            metric_type = 'ip'

        if self.metric not in ['euclidean', 'cosine', 'inner_product']:
            self.logger.warning(
                f'Invalid distance metric {self.metric} for HNSW index construction! '
                'Default to euclidean distance'
            )
            metric_type = DEFAULT_METRIC

        return metric_type

    def sync(self, delta: GENERATOR_DELTA):
        if delta is None:
            self.logger.warning('No data received in HNSW.sync. Skipping...')
            return

        for doc_id, vec_array, doc_timestamp in delta:
            idx = self._ids_to_inds.get(doc_id)

            # TODO: performance improvements possible
            # instead of creating new Ds and DAs individually
            # we can can batch
            if idx is None:
                if vec_array is None:
                    continue
                vec = vec_array.astype(HNSW_TYPE)

                self._add([vec], [doc_id])
            elif vec_array is None:
                self.delete({'ids': [doc_id]})
            else:
                vec = vec_array.reshape(1, -1).astype(HNSW_TYPE)
                da = DocumentArray(Document(id=doc_id, embedding=vec))
                self.update(da)

            if doc_timestamp > self.last_timestamp:
                self.last_timestamp = doc_timestamp

    def index_sync(self, iterator: GENERATOR_DELTA, batch_size=100) -> None:
        # there might be new operations on PSQL in the meantime
        timestamp = datetime.now(timezone.utc)
        if iterator is None:
            self.logger.warning('No data received in HNSW.sync. Skipping...')
            return

        this_batch_size = 0
        # batching
        this_batch_embeds = np.zeros((batch_size, self.dim), dtype=HNSW_TYPE)
        this_batch_ids = []

        while True:
            try:
                doc_id, vec_array, _ = next(iterator)
                if vec_array is None:
                    continue

                vec = vec_array.astype(HNSW_TYPE)
                this_batch_embeds[this_batch_size] = vec
                this_batch_ids.append(doc_id)
                this_batch_size += 1

                if this_batch_size == batch_size:
                    # do it
                    # we don't send the 0s
                    self._add(this_batch_embeds[:this_batch_size], this_batch_ids)
                    this_batch_size = 0
                    this_batch_ids = []
            except StopIteration:
                if this_batch_size > 0:
                    self._add(this_batch_embeds[:this_batch_size], this_batch_ids)
                break

        self.last_timestamp = timestamp
