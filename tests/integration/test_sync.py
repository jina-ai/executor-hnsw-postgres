import datetime
import os.path
import time

import pytest
from executor.hnswpsql import HNSWPostgresIndexer
from jina import DocumentArray, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(
    cur_dir, '..', 'docker-compose.yml'))

METRIC = 'cosine'

SHARDS = 2


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_sync(docker_compose, get_documents):
    def verify_status(f, expected_size_min):
        results = f.post('/status', None, return_responses=True)
        status_results = results[0].parameters["__results__"]
        assert len(status_results.values()) == SHARDS
        nr_hnsw_docs = sum(v['hnsw_docs'] for v in status_results.values())
        for status in status_results.values():
            psql_docs = int(status['psql_docs'])
            assert psql_docs >= expected_size_min
            assert int(nr_hnsw_docs) >= expected_size_min
        last_sync_raw = list(status_results.values())[0]['last_sync']
        last_sync_timestamp = datetime.datetime.fromisoformat(last_sync_raw)
        return nr_hnsw_docs, last_sync_timestamp

    emb_size = 10
    nr_docs_batch = 3
    nr_runs = 4

    uses_with = {'dim': emb_size, 'sync_interval': 5}
    search_docs = DocumentArray(
        get_documents(index_start=nr_docs_batch *
                      (nr_runs + 1), emb_size=emb_size)
    )

    f = Flow().add(
        uses=HNSWPostgresIndexer,
        uses_with=uses_with,
        shards=SHARDS,
        polling='all'
    )

    with f:
        nr_indexed_docs, last_sync_timestamp = verify_status(f, 0)

        for i in range(nr_runs):
            docs = get_documents(
                nr=nr_docs_batch, index_start=i * nr_docs_batch, emb_size=emb_size
            )

            f.post('/index', docs)

            got_updated_docs = False
            for _ in range(50):
                search_docs = f.post(
                    '/search', search_docs)
                assert len(search_docs[0].matches) >= nr_indexed_docs
                nr_indexed_docs, last_sync_timestamp = verify_status(
                    f, nr_indexed_docs)
                if nr_indexed_docs == (i + 1) * nr_docs_batch:
                    got_updated_docs = True
                    break
                time.sleep(0.2)
            assert got_updated_docs
