import os.path

import pytest
from executor.hnswpsql import HNSWPostgresIndexer, HnswlibSearcher, PostgreSQLStorage
from jina import DocumentArray

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '..', 'docker-compose.yml'))


def test_basic():
    indexer = HNSWPostgresIndexer(dry_run=True)
    assert isinstance(indexer._vec_indexer, HnswlibSearcher)
    assert isinstance(indexer._kv_indexer, PostgreSQLStorage)
    assert indexer._init_kwargs
    assert indexer.rebuild


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_docker(docker_compose, get_documents):
    emb_size = 10

    docs = DocumentArray(get_documents(emb_size=emb_size))

    indexer = HNSWPostgresIndexer(dim=emb_size,
                                  runtime_args={'pea_id': 0,
                                                'replica_id': 0,
                                                'parallel': 1})
    assert isinstance(indexer._vec_indexer, HnswlibSearcher)
    assert isinstance(indexer._kv_indexer, PostgreSQLStorage)
    assert indexer._init_kwargs is not None
    assert indexer.rebuild
    # test for empty sync from psql
    indexer.sync({})

    indexer.index(docs, {})

    search_docs = DocumentArray(
        get_documents(index_start=len(docs), emb_size=emb_size))
    indexer.search(search_docs, {})
    assert len(search_docs[0].matches) == 0

    indexer.sync({})
    indexer.search(search_docs, {})
    assert len(search_docs[0].matches) > 0
