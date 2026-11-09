# Data

The raw OULAD CSVs are not committed to this repo (`studentVle.csv` alone is about 450 MB).

1. Download the dataset from the Open University: https://analyse.kmi.open.ac.uk/open_dataset
2. Put these seven files in this folder:

   ```
   assessments.csv  courses.csv  studentAssessment.csv  studentInfo.csv
   studentRegistration.csv  studentVle.csv  vle.csv
   ```

Alternatively, leave them where they are and point the code at them:

```bash
export OULAD_DATA_DIR="$HOME/Downloads/OULAD Learning Analytics Dataset"
```

The code looks in `$OULAD_DATA_DIR` first, then this `data/` folder, then the repo root.
