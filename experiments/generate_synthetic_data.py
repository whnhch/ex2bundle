"""
Pre-processing: generate synthetic example snippets for the relaxation
analysis (§5.5) and scalability analysis (§5.6) experiments.

Three sampling strategies (Paper §5.6):

  --strategy single   Top-k sentences by ONE topic's score
                      (paper: "single-topic" sampling)
  --strategy random   k uniformly random sentences from the document
                      (paper: "random" sampling)
  --strategy multi    Iteratively pick a random topic, take top-5 sentences,
                      repeat until k sentences are collected
                      (paper: "multi-topic" sampling)

Output structure
----------------
   <output_dir>/<strategy>/topic<i>/synthetic_data_<k>_sentences.json

For ``random``, the ``topic<i>`` dimension is replaced by a single ``random/``
folder (no per-topic separation since the strategy is topic-agnostic).
"""

import argparse
import json
import os
import random

import pandas as pd

DOCUMENT_NAME_FIELD = "name"
NUM_COLS_BEFORE_TOPIC_SCORES = 6


def _topk_by_topic(rows, topic_idx, k):
    rows = sorted(rows, key=lambda r: r[topic_idx], reverse=True)
    return rows[:k]


def gen_single_feature(df_rows, topic_num, num_sentences):
    topic_col = topic_num + NUM_COLS_BEFORE_TOPIC_SCORES
    return _topk_by_topic(df_rows, topic_col, num_sentences)


def gen_random(df_rows, num_sentences, rng):
    shuffled = list(df_rows)
    rng.shuffle(shuffled)
    return shuffled[:num_sentences]


def gen_multi_feature(df_rows, num_sentences, n_topics, rng, chunk=5):
    selected = []
    remaining = num_sentences
    chosen_topics = []
    while remaining > 0:
        n = min(chunk, remaining)
        t = rng.randint(0, n_topics - 1)
        chosen_topics.append(t)
        topic_col = t + NUM_COLS_BEFORE_TOPIC_SCORES
        selected.extend(_topk_by_topic(df_rows, topic_col, n))
        remaining -= n
    return selected, chosen_topics


def make_record(state, doc_id, rows, topic_num=None, topics_list=None):
    rec = {
        "state_name":   state,
        "state_id":     doc_id,
        "prompt":       " ".join(str(r[4]) for r in rows),
        "sentence_ids": [r[3] for r in rows],
        "sentences":    [str(r[4]) for r in rows],
    }
    if topic_num is not None:
        rec["topic"] = topic_num
        # Use that single topic for the score column.
        topic_col = topic_num + NUM_COLS_BEFORE_TOPIC_SCORES
        rec["scores"] = [r[topic_col] for r in rows]
    if topics_list is not None:
        rec["topics"] = topics_list
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",    required=True)
    parser.add_argument("--states_file", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--strategy", choices=["single", "random", "multi", "all"],
                        default="all", help="Sampling strategy")
    parser.add_argument("--n_topics", type=int, default=10)
    parser.add_argument("--n_docs",   type=int, default=50)
    parser.add_argument("--max_len",  type=int, default=100)
    parser.add_argument("--step",     type=int, default=5)
    parser.add_argument("--seed",     type=int, default=7891)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data_csv)
    except ParserError:
        # Some datasets in this repo are semicolon-delimited.
        df = pd.read_csv(args.data_csv, sep=";")
    with open(args.states_file) as f:
        states = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(states)} states; CSV shape {df.shape}")

    rng = random.Random(args.seed)
    strategies = ["single", "random", "multi"] if args.strategy == "all" else [args.strategy]

    rows_by_state = {
        state: df[df[DOCUMENT_NAME_FIELD] == state].values.tolist()
        for state in states[: args.n_docs]
    }

    for strategy in strategies:
        print(f"\n==== strategy = {strategy} ====")

        if strategy == "single":
            for topic_num in range(args.n_topics):
                out_dir = os.path.join(args.output_dir, "single", f"topic{topic_num}")
                os.makedirs(out_dir, exist_ok=True)
                for k in range(args.step, args.max_len + 1, args.step):
                    records = []
                    for doc_id, state in enumerate(states[: args.n_docs]):
                        rows = rows_by_state[state]
                        if len(rows) < k:
                            continue
                        sel = gen_single_feature(rows, topic_num, k)
                        records.append(make_record(state, doc_id, sel, topic_num=topic_num))
                    with open(os.path.join(out_dir, f"synthetic_data_{k}_sentences.json"), "w") as f:
                        json.dump(records, f, indent=1)
                print(f"  topic{topic_num}: ✓")

        elif strategy == "random":
            out_dir = os.path.join(args.output_dir, "random")
            os.makedirs(out_dir, exist_ok=True)
            # Mirror the single-topic folder layout (one "topic" subfolder) so
            # the synthetic_runner can iterate uniformly.
            sub = os.path.join(out_dir, "topic0")
            os.makedirs(sub, exist_ok=True)
            for k in range(args.step, args.max_len + 1, args.step):
                records = []
                for doc_id, state in enumerate(states[: args.n_docs]):
                    rows = rows_by_state[state]
                    if len(rows) < k:
                        continue
                    sel = gen_random(rows, k, rng)
                    records.append(make_record(state, doc_id, sel))
                with open(os.path.join(sub, f"synthetic_data_{k}_sentences.json"), "w") as f:
                    json.dump(records, f, indent=1)
            print("  ✓")

        elif strategy == "multi":
            for topic_num in range(args.n_topics):
                out_dir = os.path.join(args.output_dir, "multi", f"topic{topic_num}")
                os.makedirs(out_dir, exist_ok=True)
                for k in range(args.step, args.max_len + 1, args.step):
                    records = []
                    for doc_id, state in enumerate(states[: args.n_docs]):
                        rows = rows_by_state[state]
                        if len(rows) < k:
                            continue
                        sel, topics = gen_multi_feature(rows, k, args.n_topics, rng)
                        records.append(make_record(state, doc_id, sel, topics_list=topics))
                    with open(os.path.join(out_dir, f"synthetic_data_{k}_sentences.json"), "w") as f:
                        json.dump(records, f, indent=1)
                print(f"  topic{topic_num}: ✓")

    print("\nDone.")


if __name__ == "__main__":
    main()
