# hri_tools

## How to install:
```
pip install git+https://github.com/Humor-Research-Institute/hri_tools.git@release-0.1.1 --upgrade
```

## How to download data
Python code:

```
import os
from hri_tools import download
os.environ["HRI_URL"] = "https://drive.google.com/file/d/1HhTgzHpruGr-_7O7wLMMxtkmjytCu-_O/view?usp=sharing"
os.environ["HRI_PASSWORD"] = "humour research 2023"
download()
```

Attention! In the public dataset, the set of 16000 one-liners has been replaced by COMB.

## Paper and citation
Data preprocessing and a detailed description of the datasets is available in the article. Please cite our article as follows:
```bibtex
@inproceedings{baranov-etal-2023-told,
    title = "You Told Me That Joke Twice: A Systematic Investigation of Transferability and Robustness of Humor Detection Models",
    author = "Baranov, Alexander  and
      Kniazhevsky, Vladimir  and
      Braslavski, Pavel",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.845",
    doi = "10.18653/v1/2023.emnlp-main.845",
    pages = "13701--13715",
}
