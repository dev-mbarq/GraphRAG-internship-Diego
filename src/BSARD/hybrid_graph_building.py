import os
import pickle
import networkx as nx
import pandas as pd


def build_hybrid_graph(G, keywords_dict, keyterm_mincount=5, unique_act_mincount=2):

    ###############
    # Parse retrieved keyword-related content and build dataframe
    ###############

    # Keep only those articles which have been already scanned
    keywords_dict_filter = {k:v for k,v in keywords_dict.items() if v != None} 

    parsed_key_terms = []

    for art in keywords_dict_filter.keys():

        for n in ["1", "2", "3", "4"]:

            parsed_key_terms.append((art, keywords_dict_filter[art][f"key_concept_{n}"]))

    # Build Dataframe
    key_terms_df = pd.DataFrame(parsed_key_terms, columns=['article_code', 'key_term'])


    ###############
    # Filter Keyterms by number of counts
    ###############

    #filtered_counts = key_terms_df["key_term"].value_counts()[key_terms_df["key_term"].value_counts() > keyterm_min_count]
    #filtered_keyterms_df = key_terms_df[key_terms_df['key_term'].isin(filtered_counts.index)]
    #filtered_keyterms_df = filtered_keyterms_df.reset_index(drop=True)


    ###############
    # Filter by lawcount
    ###############

    df_keyterms_condensed = (
        key_terms_df
        .groupby('key_term')
        .agg(
            article_codes=('article_code', list),
            count=('article_code', 'size'),
            law_count=('article_code',lambda s: s.str.split('.').str[0].nunique())
        )
        .reset_index()
    )

    df_keyterms_condensed = df_keyterms_condensed[df_keyterms_condensed['count'] > keyterm_mincount] # At least N mentions in distinc articles
    df_keyterms_condensed.reset_index(drop=True, inplace=True)
    df_keyterms_condensed_more_laws = df_keyterms_condensed[df_keyterms_condensed['law_count'] > unique_act_mincount] # At least N different laws
    df_keyterms_condensed_more_laws


    ###############
    # Add keyterms to base document graph
    ###############

    # 1) Add each key term as a new node with node_type="KeyTerm"
    G.add_nodes_from(
        (key_term, {"node_type": "KeyTerm"})
        for key_term in df_keyterms_condensed_more_laws['key_term']
    )

    # 2) Create edges between articles and key terms:
    #    article -> key_term  with relation="cites"
    #    key_term -> article  with relation="cited_in"
    G.add_edges_from(
        (article_code, key_term, {"relation": "cites"})
        for key_term, codes in zip(df_keyterms_condensed_more_laws['key_term'], df_keyterms_condensed_more_laws['article_codes'])
        for article_code in codes
    )
    G.add_edges_from(
        (key_term, article_code, {"relation": "cited_in"})
        for key_term, codes in zip(df_keyterms_condensed_more_laws['key_term'], df_keyterms_condensed_more_laws['article_codes'])
        for article_code in codes
    )

    return G


    ###############
    # Save output
    ###############

    with open(os.path.join(BSARD_data_path, 'intermediate', "hybrid_graph_full_B.pkl"), 'wb') as f:
        pickle.dump(G, f)