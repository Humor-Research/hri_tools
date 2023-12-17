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
@inproceedings{JokeTwice2023,
   title={You Told Me That Joke Twice: A Systematic Investigation of Transferability and Robustness of Humor Detection Models},
   author={Alexander Baranov, Vladimir Kniazhevsky and Pavel Braslavski},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2023}
}