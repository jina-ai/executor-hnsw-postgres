import datetime
import os.path
from collections import OrderedDict
from typing import Dict

import pytest
from executor.hnswpsql import HNSWPostgresIndexer
from jina import DocumentArray, Flow, Executor, requests
from jina.logging.profile import TimeContext

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
@pytest.mark.parametrize('nr_docs', [100])
@pytest.mark.parametrize('nr_search_docs', [10])
@pytest.mark.parametrize('emb_size', [10])
def test_replicas_integration(
    docker_compose, get_documents, nr_docs, nr_search_docs, emb_size, benchmark=False
):
    LIMIT = 10
    NR_SHARDS = 2
    NR_REPLICAS = 3
    docs = get_documents(nr=nr_docs, emb_size=emb_size)

    uses_with = {'dim': emb_size, 'limit': LIMIT, 'mute_unique_warnings': True}

    f = Flow().add(
        name='indexer',
        uses=HNSWPostgresIndexer,
        uses_with=uses_with,
        parallel=NR_SHARDS,
        replicas=NR_REPLICAS,
        # this will lead to warnings on PSQL for clashing ids
        # but required in order for the query request is sent
        # to all the shards
        polling='all',
        uses_after=MatchMerger,
    )

    with f:
        request_size = 100
        if benchmark:
            request_size = 1000

        with TimeContext(f'indexing {nr_docs}'):
            f.post('/index', docs, request_size=request_size)

        status = dict(f.post('/status', return_results=True)[0].docs[0].tags)
        assert int(status['psql_docs']) == nr_docs
        assert int(status['hnsw_docs']) == 0

        search_docs = DocumentArray(
            get_documents(index_start=nr_docs, nr=nr_search_docs, emb_size=emb_size)
        )

        if not benchmark:
            result = f.post('/search', search_docs, return_results=True)
            search_docs = result[0].docs
            assert len(search_docs[0].matches) == 0

        with TimeContext(
            f'rolling update {NR_REPLICAS} replicas x {NR_SHARDS} ' f'shards'
        ):
            f.rolling_update(pod_name='indexer', uses_with=uses_with)

        status = dict(f.post('/status', return_results=True)[0].docs[0].tags)
        assert int(status['psql_docs']) == nr_docs
        assert int(status['hnsw_docs']) >= nr_docs // NR_SHARDS

        with TimeContext(f'search with {nr_search_docs}'):
            result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) == NR_SHARDS * LIMIT


def in_docker():
    """Returns: True if running in a Docker container, else False"""
    with open('/proc/1/cgroup', 'rt') as ifh:
        if 'docker' in ifh.read():
            print('in docker, skipping benchmark')
            return True
        return False


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_benchmark_basic(docker_compose, get_documents):
    nr_docs = 1000000
    if in_docker() or ('GITHUB_WORKFLOW' in os.environ):
        nr_docs = 1000
    return test_replicas_integration(
        docker_compose=docker_compose,
        nr_docs=nr_docs,
        get_documents=get_documents,
        nr_search_docs=10,
        emb_size=128,
        benchmark=True,
    )


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_integration_cleanup(docker_compose, get_documents):
    emb_size = 10
    docs = DocumentArray(get_documents(nr=100, emb_size=emb_size))

    uses_with = {
        'dim': emb_size,
    }

    f = Flow().add(
        name='indexer',
        uses=HNSWPostgresIndexer,
        uses_with=uses_with,
    )

    with f:
        f.post('/index', docs)
        result = f.post('/status', None, return_results=True)
        result_docs = result[0].docs
        assert int(result_docs[0].tags['psql_docs']) == len(docs)

        # default to soft delete
        f.delete(docs)
        result = f.post('/status', None, return_results=True)
        result_docs = result[0].docs
        assert int(result_docs[0].tags['psql_docs']) == len(docs)

        f.post(on='/cleanup')
        result = f.post('/status', None, return_results=True)
        result_docs = result[0].docs
        assert int(result_docs[0].tags['psql_docs']) == 0
