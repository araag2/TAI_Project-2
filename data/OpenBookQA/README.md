---
configs:
- config_name: default
  data_files:
  - split: train
    path: "source/main_train.jsonl"
  - split: dev
    path: "source/main_dev.jsonl"
  - split: test
    path: "source/main_test.jsonl"
  - split: add_train
    path: "source/add_train.jsonl"
  - split: add_dev
    path: "source/add_dev.jsonl"
  - split: add_test
    path: "source/add_test.jsonl"
---

**Source Paper:** [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://arxiv.org/abs/1809.02789)

**Dataset Source:** [HuggingFace](https://huggingface.co/datasets/allenai/openbookqa)

---