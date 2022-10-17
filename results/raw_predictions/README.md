This where the raw predictions locate. 

* The directory `RF`, `molbert`, and `grover` correspond to the raw predictions from **random forest**, **MolBERT** and **GROVER**, respectively. Notably for **GROVER**, prediction results when extracted `rdkit_2d_normalized features` are used are in the subdirectory named `grover_base_rdkit` for each dataset.

* Subdirectories, `benchmark`, `cutoff6` and `reg` correspond to the results based on **benchmark datasets**, **opioids datasets (classification)** and **opioids datasets (regression)**, respectively.

* The `.csv` files store the raw predictions for each molecule in the test set, where the fold number `N` is reflected in the `test_result_foldN.csv`. Notably for **random forest**, prediction results for ECFP with different radius `r` and numBits `b` are also reflected in the filenames.