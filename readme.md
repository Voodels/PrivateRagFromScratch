Qdrant docker setup
To run Qdrant in a Docker container, use the following command:
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```
This command maps the local directory `qdrant_storage` to the Qdrant storage directory inside the container, allowing you to persist data across container restarts.