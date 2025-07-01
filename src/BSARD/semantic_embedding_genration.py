import os
import sys

import re
import pandas as pd
import networkx as nx
import tqdm
import random
import pickle
import json
import subprocess
import numpy as np
import httpx
import requests
import time

from dotenv import load_dotenv
from tqdm import tqdm


###############
# Define embedding function
###############


def assign_semantinc_node_embeddings(G, url, api_token):

    def get_ollama_embedding(input_text):

        response = requests.post(
            f"{url}/embed",
            headers={
                "Authorization": f"Bearer {api_token}",
                    },
            json={
                "input": input_text,
                }
            )
        if len(response.json()["embeddings"]) == 1:

            return response.json()["embeddings"][0]
    
        else:

            return response.json()["embeddings"]

    ###############
    # Article nodes embeddings
    ###############

    # Configuración
    BATCH_SIZE = 500

    # 1) Extraer nodos de tipo 'Article'
    article_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "Article"
    ]

    # 2) Preparar diccionario para embeddings
    embeddings_dict = {node: None for node in article_nodes}

    # 3) Generar embeddings en lotes
    pbar = tqdm(total=len(article_nodes), desc="Generating Atricle embeddings... ")
    for i in range(0, len(article_nodes), BATCH_SIZE):
        batch_nodes = article_nodes[i : i + BATCH_SIZE]
        batch_texts = [G.nodes[node]["text"] for node in batch_nodes]

        # Llamada a la API de embeddings
        batch_embeddings = get_ollama_embedding(batch_texts)

        # Asignar embeddings al diccionario
        for node, emb in zip(batch_nodes, batch_embeddings):
            embeddings_dict[node] = emb

        # Actualizar barra de progreso
        pbar.update(len(batch_nodes))

        # Pausa breve para no saturar la API
        time.sleep(0.5)

    pbar.close()

    #print("Embeddings generados y guardados en:", final_path)

    nx.set_node_attributes(G, embeddings_dict, "embedding")

    ###############
    # Keyterm node embeddings
    ###############

    # 1) Extraer nodos de tipo 'KeyTerm'
    keyterm_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "KeyTerm"
    ]

    # 2) Preparar diccionario para embeddings
    embeddings_dict = {node: None for node in keyterm_nodes}

    # 3) Generar embeddings en lotes
    pbar = tqdm(total=len(keyterm_nodes), desc="Generating KeyTerm embeddings... ")
    for i in range(0, len(keyterm_nodes), BATCH_SIZE):
        batch_nodes = keyterm_nodes[i : i + BATCH_SIZE]
        # Usamos el ID del nodo (convertido a str) como texto de entrada
        batch_texts = [str(node) for node in batch_nodes]

        # Llamada a la API de embeddings
        batch_embeddings = get_ollama_embedding(batch_texts)

        # Asignar embeddings al diccionario
        for node, emb in zip(batch_nodes, batch_embeddings):
            embeddings_dict[node] = emb

        # Actualizar barra de progreso
        pbar.update(len(batch_nodes))

        time.sleep(0.5)  # pausa breve para no saturar la API

    pbar.close()

    #print("Embeddings de KeyTerm generados y guardados en:", final_fp)

    # 5) (Opcional) Asignar el atributo 'embedding' en el grafo
    nx.set_node_attributes(G, embeddings_dict, "embedding")


    ###############
    # Initial Act node embeddings
    ###############

    # 1) Pre-count the "Act" nodes so tqdm knows the total
    act_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "Act"
    ]

    # 2) Progress bar: one tick per generated embedding
    for node, data in tqdm(act_nodes,
                           desc="Generating Act embeddings (I)...",
                           total=len(act_nodes)):      # optional: tqdm can infer it
        data["embedding"] = get_ollama_embedding(data["act_title"])
        time.sleep(0.5)


    ###############
    # Act, Book and Title node embeddings
    ###############

    # 1) Embeddings for Title: average of its Articles
    title_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Title"]
    for title in tqdm(title_nodes, desc="Generating Title embeddings... "):
        # extract only Articles with embedding
        child_articles = [
            nbr for nbr in G.successors(title)
            if G.nodes[nbr].get("node_type") == "Article"
               and "embedding" in G.nodes[nbr]
        ]
        if not child_articles:
            continue

        embs = np.stack([G.nodes[art]["embedding"] for art in child_articles])
        G.nodes[title]["embedding"] = embs.mean(axis=0).tolist()


    # 2) Embeddings for Book: average of its Articles or Titles
    book_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Book"]
    for book in tqdm(book_nodes, desc="Generating Book embeddings..."):
        # children can be Articles or Titles
        child_embs = []
        for nbr in G.successors(book):
            nt = G.nodes[nbr].get("node_type")
            if ("embedding" in G.nodes[nbr]) and nt in ("Article", "Title"):
                child_embs.append(G.nodes[nbr]["embedding"])

        if not child_embs:
            continue

        embs = np.stack(child_embs)
        G.nodes[book]["embedding"] = embs.mean(axis=0).tolist()


    # 3) Embeddings for Act: mix of its own embedding and the average of its neighbors
    act_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Act"]
    for act in tqdm(act_nodes, desc="Generating Act embeddings (II)... "):
        # neighbors with embedding (could be Book or Title depending on your graph)
        nbr_embs = [
            G.nodes[nbr]["embedding"]
            for nbr in G.successors(act)
            if "embedding" in G.nodes[nbr]
        ]
        if not nbr_embs or "embedding" not in G.nodes[act]:
            continue

        nbr_mean = np.stack(nbr_embs).mean(axis=0)
        own_emb  = np.array(G.nodes[act]["embedding"])
        # here we choose weight 0.5/0.5; adjust if you want a different balance
        mixed = (own_emb + nbr_mean) / 2
        G.nodes[act]["embedding"] = mixed.tolist()

    # 4) Embeddings for Central node: average of all its neighbors
    meta_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Central Node"]

    for meta in tqdm(meta_nodes, desc="Generating Central Node embedding... "):
        # incoming and outgoing neighbors, without duplicates
        neighs = set(G.successors(meta)).union(G.predecessors(meta))
        neigh_embs = [G.nodes[v]["embedding"] for v in neighs if "embedding" in G.nodes[v]]

        if not neigh_embs:
            continue  # no embeddings available

        embs = np.stack(neigh_embs)
        G.nodes[meta]["embedding"] = embs.mean(axis=0).tolist()

    return G

