import datetime
import os.path

import pytest
from executor.hnswpsql import HNSWPostgresIndexer
from jina import DocumentArray, Flow, Document

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '..', 'docker-compose.yml'))


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_basic_integration(docker_compose, get_documents):
    emb_size = 10

    docs = DocumentArray(get_documents(emb_size=emb_size))

    f = Flow().add(
        uses=HNSWPostgresIndexer,
        uses_with={
            'dim': emb_size
        }
    )

    with f:
        # test for empty sync from psql
        f.post('/sync', )
        result = f.post('/status', Document(), return_results=True)
        result = dict(result[0].parameters)
        status = result['last_sync']
        last_sync_timestamp = datetime.datetime.fromisoformat(
            status
        )

        f.post('/index', docs)

        search_docs = DocumentArray(
            get_documents(index_start=len(docs), emb_size=emb_size))

        result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) == 0

        f.post('/sync', )
        result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) > 0

        result = f.post('/status', Document(), return_results=True)
        result = dict(result[0].parameters)
        status = result['last_sync']
        last_sync = datetime.datetime.fromisoformat(
            status
        )
        assert last_sync > last_sync_timestamp
