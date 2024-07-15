# RAG System Using OpenAI Embedding and OpenAI Chat

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```

2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

## Create database

Create the Chroma DB.

```python
python create_database.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "[ASK YOUR QUESTION]"
```
