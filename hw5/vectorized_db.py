# your code here

import numpy as np
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class LSHDatabase:
    def __init__(self, dim: int, k: int = 6, L: int = 10) -> None:
        self.dim = dim
        self.k = k
        self.L = L

        self.random_vectors = np.random.uniform(low=-1., high=1., size=(L, k, dim))
        self.vectors = None
        self.hash_tables = [[[] for _ in range(2 ** k)] for _ in range(L)]
        self.powers_of_two = 2 ** np.arange(start=self.k - 1, stop=-1, step=-1)
        self.metadata = []

    def _compute_hashes(self, vector: np.ndarray) -> np.ndarray:
        projections = self.random_vectors @ vector  # (L, k)
        bits = (projections >= 0).astype(int)  # (L, k)
        hashes = np.sum(bits * self.powers_of_two, axis=-1)
        return hashes  # (L,)

    def _compute_hashes_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Вычисление хэшей для батча векторов"""
        projections = np.einsum('lkd,bd ->blk', self.random_vectors, vectors)  # (batch_size, L, k)
        bits = (projections >= 0).astype(np.uint8)
        hashes = np.sum(bits * self.powers_of_two, axis=-1)
        return hashes  # (batch_size, L)

    def add_vector(self, vector: np.ndarray, metadata=None) -> None:
        vector = vector.squeeze()
        self.metadata.append(metadata)

        if self.vectors is None:
            self.vectors = np.expand_dims(vector, 0)
        else:
            self.vectors = np.vstack((self.vectors, vector))
        vector_idx = len(self.vectors) - 1

        hashes = self._compute_hashes(vector)

        for hash_idx, hash_val in enumerate(hashes):
            self.hash_tables[hash_idx][hash_val].append(vector_idx)

    def add_batch_vectors(self, vectors: np.ndarray, metadata_list=None) -> None:
        if self.vectors is None:
            start_idx = 0
        else:
            start_idx = len(self.vectors)

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack((self.vectors, vectors))

        if metadata_list is None:
            metadata_list = [None] * len(vectors)
        self.metadata.extend(metadata_list)

        batch_hashes = self._compute_hashes_batch(vectors)  # (batch_size, L)

        for i in range(len(batch_hashes)):
            for hash_idx, hash_val in enumerate(batch_hashes[i]):
                self.hash_tables[hash_idx][hash_val].append(start_idx + i)

    def _lsh_candidates(self, query: np.ndarray) -> np.ndarray:
        candidate_idxs = set()
        hases = self._compute_hashes(query)

        for hash_idx, hash_val in enumerate(hases):
            candidate_idxs.update(self.hash_tables[hash_idx][hash_val])

        candidate_idxs = np.asarray(list(candidate_idxs))
        return candidate_idxs

    def search(self, query: np.ndarray, num_results: int = -1,
               use_lsh: bool = True, return_indices: bool = False) -> np.ndarray:

        if self.vectors is None:
            return np.asarray([])

        if use_lsh:
            cand_idxs = self._lsh_candidates(query)
        else:
            cand_idxs = np.arange(len(self.vectors))

        distances = []
        for idx in cand_idxs:
            dist = np.linalg.norm(query - self.vectors[idx])
            distances.append(dist)

        argsort = np.argsort(distances)
        sorted_cand_idxs = cand_idxs[argsort]
        sorted_candidates = self.vectors[sorted_cand_idxs]

        if num_results != -1:
            sorted_cand_idxs = sorted_cand_idxs[:num_results]
            sorted_candidates = sorted_candidates[:num_results]

        if return_indices:
            return sorted_candidates, sorted_cand_idxs

        return sorted_candidates


class RecursiveTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=None):
        self.chunk_size = chunk_size

        if chunk_overlap is None:
            self.chunk_overlap = chunk_size // 2
        else:
            self.chunk_overlap = chunk_overlap

        assert self.chunk_overlap < self.chunk_size

    def split_text(self, text):
        if len(text) <= self.chunk_size:
            return text
        return self._split_paragraphs(text)

    def _split_paragraphs(self, text):
        text = text.split('\n\n')
        text = [[paragraph] if len(paragraph) <= self.chunk_size else self._split_lines(paragraph) for paragraph in
                text]
        text = sum(text, [])
        return text

    def _split_lines(self, text):
        text = text.split('\n')
        text = [[line] if len(line) <= self.chunk_size else self._split_words(line) for line in text]
        text = sum(text, [])
        return text

    def _split_words(self, text):
        text = text.split()

        new_text = []
        cur_chunk = []
        for word in text:
            if len(' '.join(cur_chunk + [word])) <= self.chunk_size:
                cur_chunk += [word]
            else:
                new_text.append(' '.join(cur_chunk))
                cur_chunk += [word]

                while len(cur_chunk) > 1 and len(' '.join(cur_chunk)) > self.chunk_overlap:
                    cur_chunk = cur_chunk[1:]
        new_text.append(' '.join(cur_chunk))

        new_text = [[chunk] if len(chunk) <= self.chunk_size else self._split_symbols(chunk) for chunk in new_text]
        new_text = sum(new_text, [])

        return new_text

    def _split_symbols(self, text):
        text = list(text)

        step = self.chunk_size - self.chunk_overlap
        new_text = [''.join(text[i:self.chunk_size]) for i in range(0, len(text), step)]

        return new_text


class Embedder:
    def __init__(self, emb_model_name, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = 1024

        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
        self.model = AutoModel.from_pretrained(emb_model_name).to(self.device)
        self.model.eval()

    def average_pool(self, last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, texts: List[str]):
        prefixed_texts = [f"query: {text}" for text in texts]

        batch_dict = self.tokenizer(
            prefixed_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:2] @ embeddings[2:].T) * 100

        return embeddings.cpu().numpy()
