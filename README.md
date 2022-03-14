# 🌟 HNSW + PostgreSQL Indexer

HNSWPostgreSQLIndexer is a production-ready, scalable Indexer for the Jina neural search framework.

It combines the reliability of PostgreSQL with the speed and efficiency of the HNSWlib nearest neighbor library.

It thus provides all the CRUD operations expected of a database system, while also offering fast and reliable vector lookup.

**Requires** a running PostgreSQL database service. For quick testing, you can run a containerized version locally with:

`docker run -e POSTGRES_PASSWORD=123456  -p 127.0.0.1:5432:5432/tcp postgres:13.2`

## Syncing between PSQL and HNSW

By default, all data is stored in a PSQL database (as defined in the arguments). 
In order to add data to / build a HNSW index with your data, you need to manually call the `/sync` endpoint.
This iterates through the data you have stored, and adds it to the HNSW index.
By default, this is done incrementally, on top of whatever data the HNSW index already has.
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

### Automatic background syncing

**⚠ WARNING: Experimental feature**

Optionally, you can enable the option for automatic background syncing of the data into HNSW.
This creates a thread in the background of the main operations, that will regularly perform the synchronization.
This can be done with the `sync_interval` constructor argument, like so:

```python
Flow().add(
    uses='jinahub://HNSWPostgresIndexer',
    uses_with={'sync_interval': 5}
)
```

`sync_interval` argument accepts an integer that represents the amount of seconds to wait between synchronization attempts.
This should be adjusted based on your specific data amounts.
For the duration of the background sync, the HNSW index will be locked to avoid invalid state, so searching will be queued.
The same applies during search operations: the index is locked and indexing will be queued.

## CRUD operations

You can perform all the usual operations on the respective endpoints

- `/index`. Add new data to PostgreSQL
- `/search`. Query the HNSW index with your Documents.
- `/update`. Update documents in PostgreSQL
- `/delete`. Delete documents in PostgreSQL. 

**Note**. This only performs soft-deletion by default. 
This is done in order to not break the look-up of the Document id after doing a search. 
For a hard delete, add `'soft_delete': False'` to `parameters` of the delete request. 
You might also perform a cleanup after a full rebuild of the HNSW index, by calling `/cleanup`.

## Status endpoint

You can also get the information about the status of your data via the `/status` endpoint.
This returns a `Document` whose tags contain the relevant information.
The information can be accessed via the following keys in the `Document.tags`:

- `'psql_docs'`: number of Documents stored in the PSQL database (includes entries that have been "soft-deleted")
- `'hnsw_docs'`: the number of Documents indexed in the HNSW index
- `'last_sync'`: the time of the last synchronization of PSQL into HNSW
- `'pea_id'`: the shard number

In a sharded environment (`parallel>1`) you will get one Document from each shard. 
Each shard will have its own `'hnsw_docs'`, `'last_sync'`, `'pea_id'`, but they will all report the same `'psql_docs'`
(The PSQL database is available to all your shards).
You need to sum the `'hnsw_docs'` across these Documents, like so

```python
result_docs = f.post('/status', None, return_results=True)
total_hnsw_docs = sum(d.tags['hnsw_docs'] for d in result_docs)
```
