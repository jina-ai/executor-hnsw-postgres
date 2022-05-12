import datetime
import os.path
from collections import OrderedDict
from typing import Dict

import pytest
from executor.hnswpsql import HNSWPostgresIndexer
from jina import DocumentArray, Flow, Executor, requests
from jina.logging.profile import TimeContext

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(
    cur_dir, '..', 'docker-compose.yml'))

METRIC = 'cosine'


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_basic_integration(docker_compose, get_documents):
    emb_size = 10
    nr_docs = 299

    docs = DocumentArray(get_documents(nr=nr_docs, emb_size=emb_size))

    f = Flow().add(
        uses=HNSWPostgresIndexer,
        uses_with={
            'dim': emb_size,
        },
        shards=1,
        # this will lead to warnings on PSQL for clashing ids
        # but required in order for the query request is sent
        # to all the shards
        polling='all',
    )

    with f:
        # test for empty sync from psql
        f.post(
            '/sync',
        )
        results = f.post('/status', None, return_responses=True)
        status_results = results[0].parameters["__results__"]
        first_hnsw_docs = sum(v['hnsw_docs'] for v in status_results.values())
        for a_status in status_results.values():
            assert int(a_status['psql_docs']) == 0
        assert int(first_hnsw_docs) == 0

        status = list(status_results.values())[0]['last_sync']
        last_sync_timestamp = datetime.datetime.fromisoformat(status)

        f.post('/index', docs)

        search_docs = DocumentArray(
            get_documents(index_start=len(docs), emb_size=emb_size)
        )

        search_docs = f.post('/search', search_docs)
        assert len(search_docs[0].matches) == 0

        f.post(
            '/sync',
        )
        search_docs = f.post('/search', search_docs)
        assert len(search_docs[0].matches) > 0

        results2 = f.post('/status', None, return_responses=True)
        status_results2 = results2[0].parameters["__results__"]
        new_hnsw_docs = sum(v['hnsw_docs'] for v in status_results2.values())
        assert new_hnsw_docs > first_hnsw_docs
        assert int(new_hnsw_docs) == len(docs)
        for a_status in status_results2.values():
            assert int(a_status['psql_docs']) == len(docs)
        status2 = list(status_results2.values())[0]['last_sync']
        last_sync = datetime.datetime.fromisoformat(status2)
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
    # FIXME: rolling_update is deprecated in latest jina core, then replicas > 1 cannot pass the test.
    NR_REPLICAS = 1
    docs = get_documents(nr=nr_docs, emb_size=emb_size)

    uses_with = {'dim': emb_size, 'limit': LIMIT, 'mute_unique_warnings': True}

    f = Flow().add(
        name='indexer',
        uses=HNSWPostgresIndexer,
        uses_with=uses_with,
        shards=NR_SHARDS,
        replicas=NR_REPLICAS,
        # this will lead to warnings on PSQL for clashing ids
        # but required in order for the query request is sent
        # to all the shards
        polling='all',
        timeout_ready=-1,
    )

    with f:
        results = f.post('/status', return_responses=True)
        status_results = results[0].parameters["__results__"]
        for a_status in status_results.values():
            assert int(a_status['psql_docs']) == 0
        hnsw_docs = sum(v['hnsw_docs'] for v in status_results.values())
        assert int(hnsw_docs) == 0

        request_size = 100
        if benchmark:
            request_size = 1000

        with TimeContext(f'indexing {nr_docs}'):
            f.post('/index', docs, request_size=request_size)

        status = f.post(
            '/status', return_responses=True)[0].parameters["__results__"]
        for a_status in status.values():
            assert int(a_status['psql_docs']) == nr_docs
            assert int(a_status['hnsw_docs']) == 0

        search_docs = DocumentArray(
            get_documents(index_start=nr_docs,
                          nr=nr_search_docs, emb_size=emb_size)
        )

        if not benchmark:
            search_docs = f.post('/search', search_docs)
            assert len(search_docs[0].matches) == 0

        # NOTE: "rolling_update" is remove, refer to https://github.com/jina-ai/jina/pull/4517
        # with TimeContext(f'rolling update {NR_REPLICAS} replicas x {NR_SHARDS} shards'):
        #     f.rolling_update(deployment_name='indexer', uses_with=uses_with)

        f.post(
            '/sync',
        )

        results2 = f.post('/status', return_responses=True)
        status_results2 = results2[0].parameters["__results__"]
        for a_status in status_results2.values():
            assert int(a_status['psql_docs']) == nr_docs
        hnsw_docs2 = sum(v['hnsw_docs'] for v in status_results2.values())
        assert int(hnsw_docs2) == nr_docs

        with TimeContext(f'search with {nr_search_docs}'):
            search_docs = f.post('/search', search_docs)
        assert len(search_docs[0].matches) == NR_SHARDS * LIMIT
        # FIXME(core): see https://github.com/jina-ai/executor-hnsw-postgres/pull/7
        if benchmark:
            f.post('/clear')


def in_docker():
    """Returns: True if running in a Docker container, else False"""
    try:
        with open('/proc/1/cgroup', 'rt') as ifh:
            if 'docker' in ifh.read():
                print('in docker, skipping benchmark')
                return True
            return False
    except:
        return False


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_benchmark_basic(docker_compose, get_documents):
    docs = [1_000, 10_000, 100_000, 1_000_000]
    if in_docker() or ('GITHUB_WORKFLOW' in os.environ):
        docs.pop()
    for nr_docs in docs:
        test_replicas_integration(
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
        results = f.post('/status', None, return_responses=True)
        status_results = results[0].parameters["__results__"]
        for a_status in status_results.values():
            assert int(a_status['psql_docs']) == len(docs)

        # default to soft delete
        f.delete(docs)
        results2 = f.post('/status', None, return_responses=True)
        status_results2 = results2[0].parameters["__results__"]
        for a_status in status_results2.values():
            assert int(a_status['psql_docs']) == len(docs)

        f.post(on='/cleanup')
        results3 = f.post('/status', None, return_responses=True)
        status_results3 = results3[0].parameters["__results__"]
        for a_status in status_results3.values():
            assert int(a_status['psql_docs']) == 0


# TODO test with update. same ids, diff embeddings, assert embeddings in match has
# changed
