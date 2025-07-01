import os
import sys

import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

###############
# KEY VARIABLES
###############

# Key Inputs: bsard_corpus

# Key Parameters: N/A

# Key Outputs: G, bsard_corpus_lean


###############
# MAIN FUNCTION
###############

def build_document_graph(bsard_corpus):

    ###############
    # PARSE INFORMATION FROM BSARD MAIN TABLE
    ###############

    # Patterns for parsing the citation
    PATTERNS = {
        "Book":    re.compile(r"(Livre\s+[^\),]+)"),
        "Title":   re.compile(r"(Titre\s+[^\),]+)"),
        "Chapter": re.compile(r"(Chapitre\s+[^\),]+)"),
        "Section": re.compile(r"(Section\s+[^\),]+)"),
    }

    ###
    # Define Function to parse different fields of the original dataset
    ###

    def parse_citation(text):
        # 1) Article and Act
        article, rest = [s.strip() for s in text.split(",", 1)]
        # 2) Act (before parenthesis) and content inside ()
        act_part, *paren = rest.split("(", 1)
        act = act_part.strip()
        inside = paren[0].rstrip(")") if paren else ""
        # 3) For each field, search with its regex
        result = {
            "Article": article,
            "Act": act,
            "Book": None,
            "Title": None,
            "Chapter": None,
            "Section": None,
        }
        for key, pattern in PATTERNS.items():
            m = pattern.search(inside)
            if m:
                result[key] = m.group(1).strip()
        return result

    ###
    # Apply function to the different relevant fields
    ###

    parsed_art = []
    parsed_act = []
    parsed_book = []
    parsed_title = []
    parsed_chapter = []
    parsed_section = []

    for i in bsard_corpus['reference']:
        parsed_citation = parse_citation(i) 
        parsed_art.append(parsed_citation['Article'])
        parsed_act.append(parsed_citation['Act'])
        parsed_book.append(parsed_citation['Book'])
        parsed_title.append(parsed_citation['Title'])
        parsed_chapter.append(parsed_citation['Chapter'])
        parsed_section.append(parsed_citation['Section'])

    ###
    # Add parsed fields to the original dataset
    ###

    bsard_corpus['parsed_art'] = parsed_art
    bsard_corpus['parsed_act'] = parsed_act
    bsard_corpus['parsed_book'] = parsed_book
    bsard_corpus['parsed_title'] = parsed_title
    bsard_corpus['parsed_chapter'] = parsed_chapter
    bsard_corpus['parsed_section'] = parsed_section

    ###
    # Filter dataset
    ###

    bsard_corpus_lean = bsard_corpus[['id', 'parsed_act', 'parsed_book', 'parsed_title', 'parsed_chapter', 'parsed_section', 'parsed_art', 'article']]

    bsard_corpus_lean['parsed_book'].fillna('not_applicable', inplace=True)
    bsard_corpus_lean['parsed_title'].fillna('not_applicable', inplace=True)
    bsard_corpus_lean['parsed_chapter'].fillna('not_applicable', inplace=True)
    bsard_corpus_lean['parsed_section'].fillna('not_applicable', inplace=True)


    ###############
    # IDENTIFY ALL NODE-ENTITIES
    ###############


    # All Acts - identifying all unique acts within the dataset
    all_acts = bsard_corpus_lean['parsed_act'].unique()
    act_code_dict = {act: [i+1] for i, act in enumerate(all_acts)}
    for act in all_acts:
        tmp_act_df = bsard_corpus_lean[bsard_corpus_lean['parsed_act'] == act]

        # All Books - identifying all unique books within each act within the dataset
        all_books = tmp_act_df['parsed_book'].unique()
        book_code_dict = {book: [i+1] for i, book in enumerate(all_books)}
        book_code_dict['not_applicable'] = [0]
        for book in all_books:
            tmp_book_df = tmp_act_df[tmp_act_df['parsed_book'] == book]

            act_code_dict[act].append(book_code_dict)

            # All Titles - identifying all unique titles within all unique books within each act within the dataset
            all_titles = tmp_book_df['parsed_title'].unique()
            title_code_dict = {title: [i+1] for i, title in enumerate(all_titles)}
            title_code_dict['not_applicable'] = [0]
            for title in all_titles:
                tmp_title_df = tmp_book_df[tmp_book_df['parsed_title'] == title]

                book_code_dict[book].append(title_code_dict)

                # All Articles - identifying all unique articles within all titles within all unique books within each act within the dataset
                all_articles = tmp_title_df['parsed_art'].unique()
                article_code_dict = {article: [i+1] for i, article in enumerate(all_articles)}
                for article in all_articles:
                    tmp_article_df = tmp_title_df[tmp_title_df['parsed_art'] == article]

                    title_code_dict[title].append(article_code_dict)


    # Each entity Act, Book, Title and Article entity has now a unique identifier


    ###############
    # BUILD AND ASSIGN UNIQUE IDS TO NODE-ENTITIES
    ###############

    act_code = []
    book_code = []
    title_code = []
    chapter_code = []
    article_code = []

    for row in bsard_corpus_lean.iterrows():

        row_act_code = act_code_dict[row[1]['parsed_act']][0]
        act_code.append(str(row_act_code))

        row_book_code = act_code_dict[row[1]['parsed_act']][1][row[1]['parsed_book']][0]
        book_code.append(str(row_act_code)+'.'+str(row_book_code))

        row_title_code = act_code_dict[row[1]['parsed_act']][1][row[1]['parsed_book']][1][row[1]['parsed_title']][0]
        title_code.append(str(row_act_code)+'.'+str(row_book_code)+'.'+str(row_title_code))

        row_article_code = act_code_dict[row[1]['parsed_act']][1][row[1]['parsed_book']][1][row[1]['parsed_title']][1][row[1]['parsed_art']][0]
        article_code.append(str(row_act_code)+'.'+str(row_book_code)+'.'+str(row_title_code)+'.'+str(row_article_code))

    bsard_corpus_lean['act_code'] = act_code
    bsard_corpus_lean['book_code'] = book_code
    bsard_corpus_lean['title_code'] = title_code
    bsard_corpus_lean['article_code'] = article_code


    ###############
    # BUILD GRAPH
    ###############

    G = nx.DiGraph()

    G.add_node("Central Node", node_type="Central Node") # Add Central Node

    for act in bsard_corpus_lean['act_code'].unique():
        # Remove this filter to process all acts
        # if act != "35":
        #     continue

        df_act = bsard_corpus_lean[bsard_corpus_lean['act_code'] == act]
        act_title = df_act['parsed_act'].iloc[0]
        G.add_node(act, node_type="Act", act_title=act_title) # Add Act Node
        G.add_edge(act, "Central Node", relation="contains") # Connect Act Node to Central Node
        G.add_edge("Central Node", act, relation="belongs_to") # Connect Central Node to Act Node

        # --- Book level ---
        books = [b for b in df_act['book_code'].unique() if b != 'not_applicable']
        if books:
            for book in books:
                df_book = df_act[df_act['book_code'] == book]
                G.add_node(book, node_type="Book")
                G.add_edge(act, book, relation="contains")
                G.add_edge(book, act, relation="belongs_to")

                # --- Title level within each Book ---
                titles = [t for t in df_book['title_code'].unique() if t != 'not_applicable']
                if titles:
                    for title in titles:
                        df_title = df_book[df_book['title_code'] == title]
                        G.add_node(title, node_type="Title")
                        G.add_edge(book, title, relation="contains")
                        G.add_edge(title, book, relation="belongs_to")

                        # Articles under each Title, with "precedes"/"succeeds" links
                        prev_article = None
                        for article in df_title['article_code'].unique():
                            G.add_node(article, node_type="Article", text=df_act[df_act['article_code'] == article]['article'].iloc[0])
                            G.add_edge(title, article, relation="contains")
                            G.add_edge(article, title, relation="belongs_to")
                            if prev_article is not None:
                                G.add_edge(prev_article, article, relation="precedes")
                                G.add_edge(article, prev_article, relation="succeeds")
                            prev_article = article
                else:
                    # No Titles → Articles directly under Book
                    prev_article = None
                    for article in df_book['article_code'].unique():
                        G.add_node(article, node_type="Article", text=df_act[df_act['article_code'] == article]['article'].iloc[0])
                        G.add_edge(book, article, relation="contains")
                        G.add_edge(article, book, relation="belongs_to")
                        if prev_article is not None:
                            G.add_edge(prev_article, article, relation="precedes")
                            G.add_edge(article, prev_article, relation="succeeds")
                        prev_article = article
        else:
            # No Books → direct Title/Article level under Act
            titles = [t for t in df_act['title_code'].unique() if t != 'not_applicable']
            if titles:
                for title in titles:
                    df_title = df_act[df_act['title_code'] == title]
                    G.add_node(title, node_type="Title")
                    G.add_edge(act, title, relation="contains")
                    G.add_edge(title, act, relation="belongs_to")

                    # Articles under each Title
                    prev_article = None
                    for article in df_title['article_code'].unique():
                        G.add_node(article, node_type="Article", text=df_act[df_act['article_code'] == article]['article'].iloc[0])
                        G.add_edge(title, article, relation="contains")
                        G.add_edge(article, title, relation="belongs_to")
                        if prev_article is not None:
                            G.add_edge(prev_article, article, relation="precedes")
                            G.add_edge(article, prev_article, relation="succeeds")
                        prev_article = article
            else:
                # Neither Books nor Titles → Articles directly under Act
                prev_article = None
                for article in df_act['article_code'].unique():
                    G.add_node(article, node_type="Article", text=df_act[df_act['article_code'] == article]['article'].iloc[0])
                    G.add_edge(act, article, relation="contains")
                    G.add_edge(article, act, relation="belongs_to")
                    if prev_article is not None:
                        G.add_edge(prev_article, article, relation="precedes")
                        G.add_edge(article, prev_article, relation="succeeds")
                    prev_article = article

    return G, bsard_corpus_lean


    ###############
    # SAVE GRAPH
    ###############


    #with open(os.path.join(BSARD_data_path, 'intermediate', "base_document_graph_V2.pkl"), 'wb') as f:
    #    pickle.dump(G, f)
    #
    #bsard_corpus_lean.to_csv(os.path.join(BSARD_data_path, 'intermediate', "bsard_corpus_lean_V2.csv"))