# Tiny LM corpus

Training text for the [`tiny_transformer_lm`](../model/tiny_transformer_lm_train.py) example.

## Default: *Harry Potter and the Sorcerer's Stone* (excerpt)

The prepare script downloads **Book 1** from the community dataset
[`elricwan/HarryPotter`](https://huggingface.co/datasets/elricwan/HarryPotter)
(Hugging Face Datasets Server API) and writes a **truncated** plain-text file
so training stays fast and the repo stays small.

```bash
# from repo root
python models/examples/tiny_transformer_lm/dataset/prepare.py

# smaller / larger excerpt (characters, default 200_000 ≈ early chapters)
python models/examples/tiny_transformer_lm/dataset/prepare.py --max-chars 100000
```

Output: `harry_potter_book1.txt` (gitignored).

**Copyright:** Harry Potter text is © J.K. Rowling / publishers. Use only for
personal learning; do not redistribute generated corpora. Replace with your
own `.txt` files in this folder if you prefer.

## Custom text

Drop any `.txt` files here. Training concatenates all `*.txt` in this directory
(see `load_corpus()` in the train script).
