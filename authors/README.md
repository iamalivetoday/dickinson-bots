# Authors

One directory per writer. Each has the same shape:

```
authors/<name>/
  data/
    raw/         # original source texts (.txt) — input
    processed/   # cleaned text ready for training — generated
    README.md    # where the data came from + licensing notes
  model/         # fine-tuned weights land here (gitignored)
```

| Author     | Died | Rights        | Corpus (cleaned)                              | Chat bot |
|------------|------|---------------|-----------------------------------------------|----------|
| Dickinson  | 1886 | Public domain | 27.7k words — poems (letters excluded, editorial narration) | ✅ |
| Weil       | 1943 | Personal use  | 517k words — combined essays/notebooks        | ✅ |
| Hugo       | 1885 | Public domain | 739k words — *Les Misérables* + *Notre-Dame de Paris* | ✅ |
| Dostoevsky | 1881 | Public domain | 552k words — *Crime and Punishment* + *Brothers Karamazov* | ✅ |
| Tolstoy    | 1910 | Public domain | 911k words — *Anna Karenina* + *War and Peace* | ✅ |
| Le Guin    | 2018 | Copyrighted   | 7.5k words — 3 public speeches only            | ⚠️ not trained (corpus too thin; see `le_guin/data/README.md`) |

Word counts are the cleaned `data/processed/<name>.txt` corpora after
`scripts/prepare_data.py`. Per-bot base model and training loss live in
[`MODELS.md`](MODELS.md).

The legacy `dickinson/`, `weil/`, `jesus/` dirs at the repo root are the original
GPT-2 experiment, kept for reference. The fresh project lives under `authors/`.
