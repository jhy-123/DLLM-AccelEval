LONG_BENCH_TEMPLATE = {
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary "
        "of the report.\n\n"
        "Report:\n"
        "{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "lcc": (
        "Please complete the code given below. \n"
        "{context}Next line of code:\n"
    ),
    "lsht": (
        "请判断给定新闻的类别，下面是一些例子。\n"
        "请用中文回答。\n"
        "{context}\n"
        "{input}"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n"
        "{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", "
        "etc.\n\n"
        "The answer is: "
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\n"
        "Article: {context}\n\n"
        " Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot "
        "be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation."
        "\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or "
        "instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n"
        "{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or "
        "more sentences.\n\n"
        "Query: {input}\n"
        "Answer:"
    ),
    "repobench-p": (
        "Please complete the code given below. \n"
        "{context}{input}Next line of code:\n"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples "
        "of questions.\n\n"
        "{context}\n"
        "{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer "
        "and do not output any other words. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
}


def _render_template(dataset_name: str, doc: dict) -> str:
    template = LONG_BENCH_TEMPLATE[dataset_name]
    return template.format(context=doc.get("context", ""), input=doc.get("input", ""))


def doc_to_text_2wikimqa(doc: dict) -> str:
    return _render_template("2wikimqa", doc)


def doc_to_text_gov_report(doc: dict) -> str:
    return _render_template("gov_report", doc)


def doc_to_text_hotpotqa(doc: dict) -> str:
    return _render_template("hotpotqa", doc)


def doc_to_text_lcc(doc: dict) -> str:
    return _render_template("lcc", doc)


def doc_to_text_lsht(doc: dict) -> str:
    return _render_template("lsht", doc)


def doc_to_text_multifieldqa_en(doc: dict) -> str:
    return _render_template("multifieldqa_en", doc)


def doc_to_text_musique(doc: dict) -> str:
    return _render_template("musique", doc)


def doc_to_text_passage_retrieval_en(doc: dict) -> str:
    return _render_template("passage_retrieval_en", doc)


def doc_to_text_qasper(doc: dict) -> str:
    return _render_template("qasper", doc)


def doc_to_text_qmsum(doc: dict) -> str:
    return _render_template("qmsum", doc)


def doc_to_text_repobench_p(doc: dict) -> str:
    return _render_template("repobench-p", doc)


def doc_to_text_trec(doc: dict) -> str:
    return _render_template("trec", doc)


def doc_to_text_triviaqa(doc: dict) -> str:
    return _render_template("triviaqa", doc)
