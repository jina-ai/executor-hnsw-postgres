import os
import time

import numpy as np
import pytest
from jina import Document


@pytest.fixture()
def docker_compose(request):
    os.system(
        f'docker-compose -f {request.param} --project-directory . up  --build -d '
        f'--remove-orphans'
    )
    time.sleep(5)
    yield
    os.system(
        f'docker-compose -f {request.param} --project-directory . down '
        f'--remove-orphans'
    )


@pytest.fixture()
def get_documents():
    def get_documents_inner(nr=10, index_start=0, emb_size=7):
        random_batch = np.random.random(nr, emb_size).astype(np.float32)
        for i in range(index_start, nr + index_start):
            d = Document()
            d.id = f'aa{i}'  # to test it supports non-int ids
            d.embedding = random_batch[i - index_start]
            yield d

    return get_documents_inner


@pytest.fixture()
def runtime_args():
    return {'shard_id': 0, 'replica_id': 0, 'shards': 1}
