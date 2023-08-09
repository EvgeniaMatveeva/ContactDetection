from typing import Optional, List, Tuple, Dict


def get_contact_inds(text: str, words, min_index: int, is_contact: Dict[str, int]) -> Tuple[Optional[int], Optional[int]]:
    start = 0
    end = 0
    i = 0
    while i < len(words):
        w = words[i]
        if is_contact.get(w, 0):
            if end > 0:
                return None, None
            start = text.index(w, min_index)
            while i < len(words) and is_contact.get(words[i], 0):
                i += 1
            end = text.index(words[i - 1], start) + len(words[i - 1]) - 1

        i += 1
    if end == 0:
        return None, None
    return start, end


def split_words(words: List[str], window_size: int, overlap_size: int) -> List[str]:
    split_texts = [' '.join(words[i: i + window_size]) for i in range(0, len(words), window_size - overlap_size)]
    return split_texts
