import os
from dotenv import load_dotenv
import tiktoken
load_dotenv()


def count_embedding_tokens(text):
    # takes a list
    # Returns a list of token int and byte chunks <= max tokens allowed for OpenAI text-embedding-ada-002
    counted_tokens = list()
    tokens = list()
    encoding = tiktoken.encoding_for_model(os.getenv("OPENAI_MODEL"))
    for sentence in text:
        # Appends a list of tokens to the list() for each sentence
        tokens.append(encoding.encode(sentence))
    token_count = sum([len(lists) for lists in tokens])
    # Chunk the tokens to make the calls
    if token_count > 8191:
        chunks = int(token_count/8191)
        r = token_count - 8191 * chunks
        if token_count/8191 is int:
            chunks = chunks
        else:
            chunks += 1
        if r > 0:
            for token_list in tokens:
                list_total = sum(token_list)
                list_total += list_total
                counted_tokens.append(token_list)
                # tokens.remove(token_list)
                if list_total >= r:
                    rr = len(counted_tokens[-1])
                    counted_tokens.pop()
                    if len(counted_tokens) <= r:
                        # Find the rth place in the last token_list
                        rrr = list_total - rr
                        ith = r - rrr + 1
                        token_list_slice = token_list[:ith]
                        counted_tokens.append(token_list_slice)
                        ith -= 1
                        token_list_slice = token_list[ith:]
                        if r + len(token_list_slice) <= 8191:
                            counted_tokens.append(token_list_slice)
                            counted_tokens[-2] += counted_tokens[-1]
                            counted_tokens.pop()
                            ith = tokens.index(token_list) + 1
                            tokens[ith] = token_list_slice + tokens[ith]
                        else:
                            return counted_tokens
                    else:
                        return counted_tokens
        else:
            return tokens
        for chunk in range(chunks):
            i = r
            i += 8191
            counted_tokens.append(tokens[r:i])
            r += 8191
        return counted_tokens
    else:
        return tokens


