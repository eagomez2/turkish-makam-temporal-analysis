# About this repository
This repository contains the code used to generate the symbolic data analysis performed on the [SymbTr](https://github.com/MTG/SymbTr/) dataset used for the research project **Temporal Evolution of Makam and Usul Relationship in Turkish Makam** by Benedikt Wimmer and Esteban Gómez for the Audio and Music Processing Lab course from the Master in Sound and Music Computing at Universitat Pompeu Fabra, Barcelona.  

# Abstract
Turkish makam music is mainly rooted in an oral tradition taught in a master-apprentice setting where the pieces are learned through repetition. Its monophonic compositions feature a melodic structure called *makam* and rhythmic patterns called *usul*. Since Turkish makam features a larger number of tones per octave compared to Western music as well as a multitude of idiosyncratic rhythmic structures, analysis models that work under the structural premises of the latter are oftentimes not directly applicable to makam. The melodic progression, *seyir*, is considered an important feature to describe a makam. Previous works on computational analysis of Turkish makam utilize methods such as pitch histograms, n-gram analysis and short-time pitch histograms and are therefore based on the melodic content of a piece or analyze rhythm separately. This project explores the descriptive potential of combined temporal analysis of rhythmic pattern and melodic progression together using a subset of [SymbTr](https://github.com/MTG/SymbTr/), a symbolic dataset that contains 2200 scores with 155 different *makams* (pl. makamlar) and 88 *usuls* (pl. usuller).  

---

**IMPORTANT:** The paper is currently in peer reviewing process and therefore it is not available for download. Once this process it completed, a link for the paper will be provided. 


# How to use the repository
This repository can be used both for replicating the results presented in the paper by using the attached jupyter notebook. An explanation can be found in each cell in order to understand how to run them appropriately. A copy of the dataset is provided along with a script that downloads and uncompresses it directly from the authors repository release.

## 1. Downloading the dataset
In order to use the jupyter notebook provided in the `code` folder, it is necessary to have a working copy of the `SymbTr-2.4.3` dataset. One is already provided in the `code/dataset` folder, otherwise it can be downloaded by running the `download_dataset.py` script located in the root folder of the repository.

```bash
python download_dataset.py
```

## 2. Setting config.ini
The `config.ini` contains some paths that are used in the provided jupyter notebooks. `notation_app_path` is the location of the notation software of choice in your computer that is used to display scores. [Muse Score](https://musescore.org/) is highly recommended as it is free and cross-platform. Additionally, `music_xml_path` is the path to the MusicXML files used for the analysis. By default it is stored inside the dataset's folder in `dataset/SymbTr-2.4.3/MusicXML` but you can adapt it to your own needs.

## 3. Notebook and pre-computed data
Finally, `dataset_explorer.ipynb` notebook in the `code` folder contains all the necessary cells to generate the analysis and plot used in the paper directly from the dataset files. The selecion of usuller and makamlar can be changed to computer the same parameters for other usuller and makamlar. **Some cells may take several minutes to complete** and for this reason, precomputed plots are provided in the `precomputed_plots` folder.

---

# Structure of the repository  
The steps 1 - 3 should contain all the necessary information to explore the data, but here a further explanation is provided in case any code modifications are needed in order to adapt the content to a different scenario.

- `code`:  This folder contains the necessary notebooks and custom packages to generate the plots shown in the paper as well as to explore the dataset.  
  - `config.ini`: Configuration file with some necessary variables described in (2).
  - `dataset`: Folder containing the SymbTr-2.4.3 dataset.
  - `dataset_explorer.ipynb`: Main notebook used to analyse data and generate the plots presented in the paper.
  - `packages`: A series of classes defined to manipulate the dataset content. Additionally, some JSON and .musicxml files that contain makam and usul definitions are provided.
- `download_dataset.py`: Script to download the SymbTr-2.4.3 dataset and put it in the default location.
- `paper`: Resulting paper (to be added after peer reviewing).
- `precomputed_plots`: Folder containing pre-computed plots shown in the paper. The naming convention of each file is explained in `dataset_explorer.ipynb`.    
- `slides`: This folder contains the slides used for the research topic proposal.