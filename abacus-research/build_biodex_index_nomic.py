# =============== NEW FEATURE ===============
"""
Reconstroi o indice Chroma do BioDEX com embeddings LOCAIS (nomic via vLLM).

Por que: o indice publicado pelo upstream (abacus-data.tar.gz) usa text-embedding-3-small
(dim 1536). O biodex-demo1.py consulta com nomic-embed-text-v1 (dim 768) -> dimensoes
incompativeis -> o retrieve falha e devolve listas vazias SEM erro.

Fonte dos termos (uma das duas):
  --terms-file reaction_terms.txt          (1 termo por linha; formato do helper upstream)
  --from-path .chroma-biodex-openai        (le os documentos de um indice ja existente)

Uso (com o servidor de embeddings no ar, porta 8001 do vllm-axia.sh):
  mv .chroma-biodex .chroma-biodex-openai
  python build_biodex_index_nomic.py --from-path .chroma-biodex-openai --out-path .chroma-biodex
"""
import argparse
import os
import sqlite3
import sys

import chromadb
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction


def terms_from_sqlite(path: str) -> list[str]:
    """
    Le os termos direto do chroma.sqlite3, sem usar a API do chromadb.
    Serve quando o indice de origem foi criado por outra versao do chromadb e a API
    se recusa a abri-lo. O Chroma guarda o texto de cada item em embedding_metadata
    com a chave 'chroma:document'.
    """
    db = os.path.join(path, "chroma.sqlite3")
    if not os.path.isfile(db):
        raise FileNotFoundError(db)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT string_value FROM embedding_metadata "
            "WHERE key = 'chroma:document' AND string_value IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def load_terms(args) -> list[str]:
    if args.terms_file:
        with open(args.terms_file) as f:
            terms = [line.strip() for line in f if line.strip()]
    else:
        try:
            src = chromadb.PersistentClient(args.from_path).get_collection(args.collection)
            terms, offset, page = [], 0, 5000
            while True:
                batch = src.get(limit=page, offset=offset, include=["documents"])["documents"]
                if not batch:
                    break
                terms.extend(batch)
                offset += len(batch)
        except Exception as e:
            print(f"[aviso] API do chromadb nao abriu {args.from_path} ({e!r}); lendo o SQLite direto")
            terms = terms_from_sqlite(args.from_path)
    terms = [t.strip() for t in terms if t and t.strip()]
    if not terms:
        sys.exit("[erro] nenhum termo encontrado na fonte")
    # remove duplicatas preservando ordem
    return list(dict.fromkeys(terms))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--terms-file")
    g.add_argument("--from-path")
    p.add_argument("--out-path", default=".chroma-biodex")
    p.add_argument("--collection", default="biodex-reaction-terms")
    p.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1")
    p.add_argument("--embed-api-base", default="http://localhost:8001/v1")
    p.add_argument("--batch-size", default=512, type=int)
    args = p.parse_args()

    if args.from_path and args.from_path == args.out_path:
        sys.exit("[erro] --from-path e --out-path iguais; mova o indice antigo antes.")

    terms = load_terms(args)
    print(f"{len(terms)} termos carregados")

    ef = OpenAIEmbeddingFunction(api_key="EMPTY", model_name=args.embed_model, api_base=args.embed_api_base)
    client = chromadb.PersistentClient(args.out_path)
    try:
        client.delete_collection(args.collection)  # reconstrucao limpa e deterministica
    except Exception:
        pass
    col = client.create_collection(
        name=args.collection,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # search_func assume distancia de cosseno
    )

    for start in range(0, len(terms), args.batch_size):
        batch = terms[start:start + args.batch_size]
        col.add(
            documents=batch,
            embeddings=ef(batch),
            ids=[f"id{i}" for i in range(start, start + len(batch))],
        )
        print(f"  {min(start + args.batch_size, len(terms))}/{len(terms)}", end="\r")

    print(f"\nOK: {col.count()} termos em {args.out_path}/{args.collection}")
    print("probe('headache'):", col.query(query_embeddings=ef(["headache"]), n_results=5)["documents"][0])
# =============== END NEW FEATURE ===============