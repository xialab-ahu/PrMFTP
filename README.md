# MLTP
multi-label therapeutic peptides prediction based on multi-head self-attention mechanism and class weights optimization


## Introduction
Motivation: 

The peptide drugs has wide applications, such as lowering blood glucose levels, blood pressure, reducing inflammation, and resistance diseases. However, it is exceedingly expensive and time-consuming for experiments to determine the function of numerous peptides. Therefore, computational methods such as machine learning are becoming more and more important for peptide function prediction. Meanwhile, there are finitude research about applying the deep learning method to address multi-label therapeutic peptide prediction. In this study, we develop a deep learning approach that determining the peptide function.

Results: 

We propose a multi-label classification model, named MLTP, to predict peptide function. In the field of functional peptide prediction, compared with state-of-the-art single-label predictor, MLTP can predict multiple functions including AAP, anti-angiogenic peptide、ABP, anti-bacterial peptide、 ACP, Anti-cancer peptide、 ACVP, Anti-coronavirus peptide、 ADP, Anti-diabetic peptide、 AEP, Anti-endotoxin peptide、 AFP, Anti-fungal peptide、 Anti-HIV peptide, AHIVP、 AHP, Anti-hypertensive peptide、 AIP, Anti-inflammatory peptide、 AMRSAP, Anti-MRSA peptide、 APP, Anti-parasitic peptide、 ATP, Anti-tubercular peptide、 AVP, Anti-viral peptide、 BBP, Blood-brain Barrier peptide、 BIP, Biofilm-inhibitory peptide、 CPP, Cell-penetrating peptide、 DPPIP, Dipeptidyl Peptidase IV peptide、 QSP, Quorum-sensing peptide、 SBP, Surface-binding peptide、 THP, Tumor-homing peptide simultaneously. Meanwhile, our model takes the raw sequence vector as input to replace biological features encoding from peptide sequences, extracts convolution and sequence features from the raw sequence, and combines with recurrent neural network to extract global features of sequences, than the multi head self attention mechanism is used to optimize the extracted features, so as to provide the prediction sequence of the model, and the class weight is used to solve the problem of data imbalance. The validation experiments conducted on the dataset show that MLTP has superior prediction performance. 

![draft](./figures/MLTP.jpg)


## Related Files

#### MLBP

| FILE NAME           | DESCRIPTION                                                  |
| :------------------ | :----------------------------------------------------------- |
| main.py             | the main file of MLTP predictor (include data reading, encoding, and data partitioning) |
| train.py            | train model |
| model.py            | model construction |
| test.py             | test model result |
| evaluation.py       | evaluation metrics (for evaluating prediction results) |
| dataset             | data         |
| model               | models of MLTP           |
| result               | result of MLTP           |


## Installation
- Requirement
  
  OS：
  
  - `Windows` ：Windows10 or later
  
  - `Linux`：Ubuntu 16.04 LTS or later
  
  Python：
  
  - `Python` >= 3.6
  
- Download `MLTP`to your computer

  ```bash
  git clone https://github.com/xialab-ahu/PrMFTP.git
  ```

- open the dir and install `requirement.txt` with `pip`

  ```
  cd PrMFTP
  pip install -r requirement.txt
  ```


## Run MLTP on a new test fasta file
```shell
python predictor.py --file test.fasta --out_path result
```

- `--file` : input the test file with fasta format

- `--out_path`: the output path of the predicted results


## Contact
Please feel free to contact us if you need any help.

