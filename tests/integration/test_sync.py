import datetime
import os.path
import time

import pytest
from executor.hnswpsql import HNSWPostgresIndexer
from jina import DocumentArray, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '..', 'docker-compose.yml'))

METRIC = 'cosine'


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_sync(docker_compose, get_documents):
    emb_size = 10
    nr_docs = 299

    uses_with = {'dim': emb_size, 'auto_sync': True}

    docs = DocumentArray(get_documents(nr=nr_docs, emb_size=emb_size))

    f = Flow().add(
        uses=HNSWPostgresIndexer,
        uses_with=uses_with,
    )

    with f:
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

        time.sleep(10)  # wait for syncing
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
