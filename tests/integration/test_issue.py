import numpy as np
from jina import DocumentArray, Flow, Document
from executor.hnswpsql import HNSWPostgresIndexer


def test_issue():
    def get_documents(nr=10, index_start=0, emb_size=7):
        random_batch = np.random.random([nr, emb_size]).astype(np.float32)
        for i in range(index_start, nr + index_start):
            d = Document()
            d.id = f'aa{i}'  # to test it supports non-int ids
            d.embedding = random_batch[i - index_start]
            yield d


    emb_size = 10

    docs = DocumentArray(get_documents(emb_size=emb_size))

    with Flow().add(
            uses=HNSWPostgresIndexer,
            uses_with={
                'dim': emb_size
            },
            parallel=3,
            polling='all',
            install_requirements=True,
            uses_after='jinahub+docker://MatchMerger/v0.2'
    ) as f:
        # test for empty sync from psql
        result = f.post('/status', None, return_results=True)
        result = dict(result[0].parameters)
        print(result)
        f.post('/index', docs)
        f.post('/sync')
        search_docs = DocumentArray(
            get_documents(index_start=len(docs), emb_size=emb_size))

        result = f.post('/search', search_docs, return_results=True)
        search_docs = result[0].docs
        assert len(search_docs[0].matches) > 0

        result = f.post('/status', None, return_results=True)
        result = dict(result[0].parameters)
        print(f'## result = {result}')
