* The processed benchmark datasets with two columns (`SMILES`, `label`) are listed as follows. Among them, BACE, BBBP and HIV are for classification task whereas ESOL, FreeSolv and Lipop are for regression task. <br>
    * `BACE_benchmark.csv`
    * `BBBP_benchmark.csv`
    * `HIV_benchmark.csv`
    * `ESOL_benchmark.csv`
    * `FreeSolv_benchmark.csv`
    * `Lipop_benchmark.csv`

* The directory `random_split/benchmark` has the 30-fold training/validation/test sets under **random-split** scheme whereas the directory `scaffold_split/benchmark` has the 30-fold training/validation/test sets under **scaffold-split** scheme.

* The directory `rdkit_ftrs` is where the extracted `rdkit_2d_normalized features` locate, which are used in **GROVER_RDKit**. 