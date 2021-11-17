import copy
import datetime
import os.path
from collections import OrderedDict
from typing import Dict

import pytest
from executor.hnswpsql import HNSWPostgresIndexer
from jina import DocumentArray, Flow, Executor, requests

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '..', 'docker-compose.yml'))

METRIC = 'cosine'


class MatchMerger(Executor):
    @requests(on='/search')
    def merge(self, docs_matrix, parameters: Dict, **kwargs):
        if docs_matrix:
            results = OrderedDict()
            for docs in docs_matrix:
                for doc in docs:
                    if doc.id in results:
                        results[doc.id].matches.extend(doc.matches)
                    else:
                        results[doc.id] = doc

            top_k = parameters.get('top_k')
            if top_k:
                top_k = int(top_k)

            for doc in results.values():
                doc.matches = sorted(doc.matches, key=lambda m: m.scores[METRIC].value)[
                    :top_k
                ]

            docs = DocumentArray(list(results.values()))
            return docs


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_basic_integration(docker_compose, get_documents):
    emb_size = 10

    docs = DocumentArray(get_documents(emb_size=emb_size))

    f = Flow().add(
        uses=HNSWPostgresIndexer,
        uses_with={'dim': emb_size},
        parallel=3,
        # this will lead to warnings on PSQL for clashing ids
        # but required in order for the query request is sent
        # to all the shards
        polling='all',
        uses_after=MatchMerger,
    )

    with f:
        # test for empty sync from psql
        f.post(
            '/sync',
        )
        result = f.post('/status', None, return_results=True)
        result_docs = result[0].docs
        first_hnsw_docs = sum(d.tags['hnsw_docs'] for d in result_docs)
        assert int(result_docs[0].tags['psql_docs']) == 0
        assert int(first_hnsw_docs) == 0

        status = result_docs[0].tags['last_sync']
        last_sync_timestamp = datetime.datetime.fromisoformat(status)

        f.post('/index', docs)

        search_docs = DocumentArray(
            get_documents(index_start=len(docs), emb_size=emb_size)
        )

        result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) == 0

        f.post(
            '/sync',
        )
        result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) > 0

        result = f.post('/status', None, return_results=True)
        result_docs = result[0].docs
        new_hnsw_docs = sum(d.tags['hnsw_docs'] for d in result_docs)
        assert new_hnsw_docs > first_hnsw_docs
        assert int(new_hnsw_docs) == len(docs)
        assert int(result_docs[0].tags['psql_docs']) == len(docs)
        status = result_docs[0].tags['last_sync']
        last_sync = datetime.datetime.fromisoformat(status)
        assert last_sync > last_sync_timestamp


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_replicas_integration(docker_compose, get_documents):
    emb_size = 10
    LIMIT = 10
    NR_SHARDS = 2
    docs = DocumentArray(get_documents(nr=100, emb_size=emb_size))

    uses_with = {
        'dim': emb_size,
        'limit': LIMIT,
    }

    f = Flow().add(
        name='indexer',
        uses=HNSWPostgresIndexer,
        uses_with=uses_with,
        parallel=NR_SHARDS,
        replicas=3,
        # this will lead to warnings on PSQL for clashing ids
        # but required in order for the query request is sent
        # to all the shards
        polling='all',
        uses_after=MatchMerger,
    )

    with f:
        f.post('/index', docs)

        search_docs = DocumentArray(
            get_documents(index_start=len(docs), emb_size=emb_size)
        )

        result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) == 0

        uses_with = copy.deepcopy(uses_with)
        uses_with['startup_sync'] = True

        f.rolling_update(pod_name='indexer', uses_with=uses_with)
        result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) == NR_SHARDS * LIMIT
