from student.tokenizer_utils import *
from student.pretokenization_example import *
# from tokenizer_utils import *
# from pretokenization_example import *
from collections import Counter
from tqdm import tqdm
import json
from collections.abc import Iterable, Iterator
import torch

def bpe_pretokenize_all(file_path, number_of_processes, special_tokens):
    with open(file_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, number_of_processes, b"<|endoftext|>")
        pretok_dicts = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            text_chunk = file.read(end - start).decode("utf-8", errors="ignore")
            pretok_dict = regex_pretokenize_to_dict_chunk(text_chunk, special_tokens)
            pretok_dicts.append(pretok_dict)
        pretok_dict = merge_dicts(pretok_dicts)
    return pretok_dict

def compute_pairwise_counts(current_toks):
    pairwise_counts = Counter()
    for key, value in current_toks.items():
        for i in range(0, len(key) - 1):
            pairwise_counts[(key[i], key[i + 1])] += value
    return pairwise_counts

def apply_single_merge(bytes_tuple, merge):
    output = []
    i = 0
    while i < len(bytes_tuple):
        if (
            i < len(bytes_tuple) - 1
            and bytes_tuple[i] == merge[0]
            and bytes_tuple[i + 1] == merge[1]
        ):
            output.append(bytes_tuple[i] + bytes_tuple[i + 1])
            i += 2
        else:
            output.append(bytes_tuple[i])
            i += 1
    return output

def bpe_train(input_path, special_tokens, vocab_size=1000, number_of_processes=4):
    pretok_dict = bpe_pretokenize_all(input_path, number_of_processes, special_tokens)
    byte_pretok_dict = {}
    for key, value in pretok_dict.items():
        key_bytes = tuple([bytes([b]) for b in key])
        byte_pretok_dict.update({key_bytes: value})
    
    merges = []
    vocab = {}
    for tok in special_tokens:
        vocab[len(vocab.keys())] = tok.encode("utf8")
    for i in range(0, 256):
        vocab[len(vocab.keys())] = bytes([i])

    latest_bytified_dict = byte_pretok_dict
    counts = compute_pairwise_counts(latest_bytified_dict)
    pbar = tqdm(total=vocab_size - len(vocab))
    while len(vocab) < vocab_size and counts:
        pbar.update(1)
        top_count = counts.most_common(1)[0][1]
        all_top_keys = [key for key in counts.keys() if counts[key] >= top_count]
        best_lex_key = max(all_top_keys)
        merges.append(best_lex_key)
        vocab[len(vocab)] = best_lex_key[0] + best_lex_key[1]

        new_bytified_dict = {}
        for key, value in latest_bytified_dict.items():
            merged = apply_single_merge(key, best_lex_key)
            key_toks = tuple(merged)
            new_bytified_dict[key_toks] = new_bytified_dict.get(key_toks, 0) + value
        latest_bytified_dict = new_bytified_dict

        counts = compute_pairwise_counts(latest_bytified_dict)
    pbar.close()
    return vocab, merges

class Tokenizer(object):
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens is not None:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tok_re = "(" + "|".join([re.escape(tok) for tok in sorted_special_tokens]) + ")"
        else:
            self.special_tok_re = ""
        self.bytes_to_id = {byte: tid for tid, byte in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = [tuple(line.split()) for line in f]
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens is not None:
            sub_chunks = re.split(self.special_tok_re, text)
        else:
            sub_chunks = [text]
        all_toks = []
        for sub_chunk in sub_chunks:
            if self.special_tokens is not None and sub_chunk in self.special_tokens:
                all_toks += [self.bytes_to_id[sub_chunk.encode("utf8")]]
            else:
                for chunk in regex_pretokenize_to_dict(sub_chunk):
                    all_toks += self.get_token_from_chunk(chunk)
        return all_toks

    def get_token_from_chunk(self, chunk):
        if chunk in self.bytes_to_id:
            return [self.bytes_to_id[chunk]]

        # bytified_tok = list([bytes([b]) for b in chunk])
        # ans = []
        # i = 0
        # while i < len(bytified_tok):
        #     j = i + 1
        #     pivot = j
        #     while j < len(bytified_tok):
        #         if b"".join(bytified_tok[i:j+1]) not in self.bytes_to_id:
        #             if j+1 - i > 30:
        #                 break
        #             j += 1
        #         else:
        #             j += 1
        #             pivot = j
        #     ans.append(self.bytes_to_id[b"".join(bytified_tok[i:pivot])])
        #     i = pivot
        # return ans

        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        symbols = [bytes([b]) for b in chunk]
        if not symbols:
            return []

        while True:
            best_i = None
            best_rank = None

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_i = i

            if best_i is None:
                break

            symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]

        return [self.bytes_to_id[s] for s in symbols]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for sentence in iterable:
            ids = self.encode(sentence)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        tok = b"".join([self.vocab[id] for id in ids])
        return tok.decode("utf8", errors="ignore")

def __main__():
    vocab, merges = bpe_train("tests/fixtures/tinystories_sample_5M.txt", ["<|endoftext|>"], 500, 4)
    print(vocab)
    print(merges)

if __name__ == "__main__":
    __main__()