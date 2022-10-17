* The processed opioids-related datasets with two columns (`SMILES`, `label`) are listed as follows. The suffix, `cutoff6` or `reg`, in the `.csv` file names indicate the task setting, i.e., classification or regression. Among them, MDR1, CYP2D6 and CYP3A4 are related to the pharmacokinetic perspective of opioid overdose while MOR, DOR and KOR related to its pharmacodynamic perspective. <br>
    * `MDR1_cutoff6.csv` & `MDR1_reg.csv`
    * `CYP2D6_cutoff6.csv` & `CYP2D6_reg.csv`
    * `CYP3A4_cutoff6.csv` & `CYP3A4_reg.csv`
    * `MOR_cutoff6.csv` & `MOR_reg.csv`
    * `DOR_cutoff6.csv` & `DOR_reg.csv`
    * `KOR_cutoff6.csv` & `KOR_reg.csv`

* The directory `random_split` has the 30-fold training/validation/test sets under **random-split** scheme whereas the directory `scaffold_split` has the 30-fold training/validation/test sets under **scaffold-split** scheme. Again, `cutoff6` and `reg` in the subdirectory name reflect the task setting. 

* The directory `rdkit_ftrs` is where the extracted `rdkit_2d_normalized features` locate, which are used in **GROVER_RDKit**. 