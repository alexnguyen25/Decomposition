# Vendored third-party code — BEATs

Source: https://github.com/microsoft/unilm/tree/master/beats
Copyright (c) 2022 Microsoft. Licensed under the MIT License.
Paper: *BEATs: Audio Pre-Training with Acoustic Tokenizers*, https://arxiv.org/abs/2212.09058

`BEATs.py`, `backbone.py` and `modules.py` are vendored verbatim except for one
change: the flat imports (`from backbone import ...`, `from modules import ...`)
were rewritten as package-relative (`from .backbone import ...`) so this works as
an ordinary Python package instead of requiring the directory on `sys.path`.

The pretrained checkpoint (`BEATs_iter3_plus_AS2M.pt`, 345 MB) is **not** vendored.
`scripts/fetch_models.py` downloads it from the Hugging Face Hub.

Only the backbone is third-party. The 20-way classification head on top of these
embeddings was trained for this project (see `docs/research/`).
