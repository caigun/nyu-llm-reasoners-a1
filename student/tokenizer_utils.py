import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def regex_pretokenize(text):
    return re.finditer(PAT, text)

def regex_pretokenize_to_dict_chunk(text, special_tokens):
    pretok_dict = {}
    special_tok_re = "|".join([re.escape(tok) for tok in special_tokens])
    sub_chunks = re.split(special_tok_re, text)
    for sub_chunk in sub_chunks:
        for match in regex_pretokenize(sub_chunk):
            this_word = sub_chunk[match.start():match.end()]
            if len(this_word) > 0 and not this_word in special_tokens:
                pretok_dict[this_word.encode("utf8")] = pretok_dict.get(this_word.encode("utf8"), 0) + 1
    return pretok_dict

def regex_pretokenize_to_dict(text):    # for encoding
    tokens = []
    for match in regex_pretokenize(text):
        this_word = text[match.start():match.end()]
        if len(this_word) > 0:
            tokens.append(this_word.encode("utf8"))
    return tokens

def merge_dicts(dicts):
    merged_dict = {}
    for dict in dicts:
        for key, value in dict.items():
            merged_dict[key] = merged_dict.get(key, 0) + value
    return merged_dict


def __main__():
    text = b"This is a test".decode("utf-8", errors="ignore")
    matchs = regex_pretokenize(text)
    for match in matchs:
        this_word = text[match.start():match.end()]
        print(this_word)

if __name__ == "__main__":
    __main__()