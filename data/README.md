# Data files

| File | Rows | Columns | What it is |
|---|---|---|---|
| `iris.csv` | 150 | 4 measurements + class (0 setosa, 1 versicolor, 2 virginica) | Fisher's Iris flowers, from the UCI Machine Learning Repository |
| `digits8x8.csv` | 1797 | 64 pixels (0 to 16) + digit (0 to 9) | The UCI "optical recognition of handwritten digits" **test set** (8x8 pictures) |

Both files have a header line; the **last column is the class**. `datasets.py` divides the digit pixels by 16 so they lie between 0 and 1.

## Where they came from

Both files were exported **once** from the copies bundled with scikit-learn, only to get plain CSV files. scikit-learn is **not** used anywhere in this project, is not in `requirements.txt`, and a test checks that no source file imports it.

* Iris: UCI Machine Learning Repository (R. A. Fisher, 1936), licence CC BY 4.0 as listed by UCI (check the UCI page if you redistribute). The scikit-learn copy differs from the UCI file in two rows (samples 35 and 38), where it follows the numbers printed in Fisher's paper.
* Digits: UCI optdigits (E. Alpaydin, C. Kaynak, Bogazici University), licence CC BY 4.0 as listed by UCI (check the UCI page if you redistribute).

Please cite the original sources if you publish anything based on them.
