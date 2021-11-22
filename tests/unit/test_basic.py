import datetime
import os.path

import pytest
from executor.hnswpsql import HNSWPostgresIndexer, HnswlibSearcher, PostgreSQLStorage
from jina import DocumentArray

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '..', 'docker-compose.yml'))


def test_basic(runtime_args):
    indexer = HNSWPostgresIndexer(
        dry_run=True, startup_sync=False, runtime_args=runtime_args
    )
    assert isinstance(indexer._vec_indexer, HnswlibSearcher)
    assert isinstance(indexer._kv_indexer, PostgreSQLStorage)
    assert indexer._init_kwargs is not None
    status = dict(indexer.status()[0].tags)
    assert status['psql_docs'] is None
    assert status['hnsw_docs'] == 0.0  # protobuf converts ints to floats
    assert datetime.datetime.fromisoformat(status['last_sync']) == datetime.datetime.min


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_docker(docker_compose, get_documents, runtime_args):
    emb_size = 10

    docs = DocumentArray(get_documents(emb_size=emb_size))

    indexer = HNSWPostgresIndexer(dim=emb_size, runtime_args=runtime_args)
    assert isinstance(indexer._vec_indexer, HnswlibSearcher)
    assert isinstance(indexer._kv_indexer, PostgreSQLStorage)
    assert indexer._init_kwargs is not None
    # test for empty sync from psql
    indexer.sync({})

    indexer.index(docs, {})

    search_docs = DocumentArray(get_documents(index_start=len(docs), emb_size=emb_size))
    indexer.search(search_docs, {})
    assert len(search_docs[0].matches) == 0

    indexer.sync({})
    indexer.search(search_docs, {})
    assert len(search_docs[0].matches) > 0
