The results directory contains 

* `raw_predictions`: raw predictions from **random forest**, **MolBERT** and **GROVER** 

* `processed_performance`: **individual performance metrics** calculated from raw predictions (`grand_perf_df` in filenames) and **aggregated performance metrics** (mean & standard deviation; `agg_perf_df` in filenames)

* `stats`: statistical analysis results with **p values from pairwise comparisons**

* `structures`: **scaffolds frequency & structural traits** in subdirectories `benchmark` & `opioids`, visualized **top 30 scaffolds** in subdirectory `scaffolds` and visualized **AC showcase molecules** in subdirectory `ACs`