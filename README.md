# 🌟 HNSW + PostgreSQL Indexer

HNSWPostgreSQLIndexer Jina is a production-ready, scalable Indexer for the Jina neural search framework.

It combines the reliability of PostgreSQL with the speed and efficiency of the HNSWlib nearest neighbor library.

It thus provides all the CRUD operations expected of a database system, while also offering fast and reliable vector lookup.

**Requires** a running PostgreSQL database service. For quick testing, you can run a containerized version locally with:

`docker run -e POSTGRES_PASSWORD=123456  -p 127.0.0.1:5432:5432/tcp postgres:13.2`

## Syncing between PSQL and HNSW

By default, all data is stored in a PSQL database (as defined in the arguments). 
In order to add data to / build a HNSW index with your data, you need to manually call the `/sync` endpoint.
This iterates through the data you have stored, and adds it to the HNSW index.
By default, this is done incrementally. 
If you want to completely rebuild the index, use the parameter `rebuild`, like so:

```python
flow.post(on='/sync', parameters={'rebuild': True})
```

At start-up time, the data from PSQL is synced into HNSW automatically.
You can disable this with: 

```python
Flow().add(
    uses='jinahub://HNSWPostgresIndexer',
    uses_with={'startup_sync': False}
)
```

## CRUD operations

You can perform all the usual operations on the respective endpoints

- `/index`. Add new data to PostgreSQL
- `/search`. Query the HNSW index with your Documents.
- `/update`. Update documents in PostgreSQL
- `/delete`. Delete documents in PostgreSQL. 

**Note**. This only performs soft-deletion by default. 
This is done in order to not break the look-up of the document id after doing a search. 
For a hard delete, add `'soft_delete': False'` to `parameters`. 
You might also perform a cleanup after a full rebuild of the HNSW index, by calling `/cleanup`.
