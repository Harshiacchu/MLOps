## Data Version Control (DVC)

- [DVC](https://dvc.org/) is an open-source tool that serves as a powerful asset in the machine learning project toolkit, with a primary focus on data versioning.
- **Data versioning** is a critical aspect of any ML project. It allows you to track changes and updates in your datasets over time, ensuring you can always recreate, compare, and reference specific dataset versions used in your experiments.
- In this lab tutorial, we will be utilizing DVC with Google Cloud Storage to enhance data versioning capabilities, ensuring efficient data management and collaboration within your machine learning project.
### Creating a Google Cloud Storage Bucket
1. Navigate to [Google Cloud Console](https://console.cloud.google.com/).
2. Ensure you've created a new project specifically for this lab.
3. In the Navigation menu, select "Cloud Storage," then go to "Buckets," and click on "Create a new bucket."
4. Assign a unique name to your bucket.
5. Select the region as `us-east1`
6. Proceed by clicking "Continue" until your new bucket is successfully created.
7. Once the bucket is created, we need to get the credentials to connect the GCP remote to the project. Go to the `IAM & Admin` service and go to `Service Accounts` in the left sidebar.
8. Click the `Create Service Account` button to create a new service account that you'll use to connect to the DVC project in a bit. Now you can add the name and ID for this service account and keep all the default settings. We've chosen `lab2` for the name. Click `Create and Continue` and it will show the permissions settings. Select `Owner` in the dropdown and click `Continue`.
9. Then add your user to have access to the service account and click `Done`. Finally, you'll be redirected to the `Service accounts` page. You’ll see your service account and you’ll be able to click on `Actions` and go to where you `Manage keys` for this service account. 
10. Once you’ve been redirected, click the `Add Key` button and this will bring up the credentials you need to authenticate your GCP account with your project. Proceed by downloading the credentials in JSON format and securely store the file. This file will serve as the authentication mechanism for DVC when connecting to Google Cloud.
### Installing DVC with Google Cloud Support
- Ensure you have DVC with Google Cloud support installed on your system by using the following command:
	`pip install dvc[gs]`
- Note that, depending on your chosen [remote storage](https://dvc.org/doc/user-guide/data-management/remote-storage), you may need to install optional dependencies such as `[s3]`, `[azure]`, `[gdrive]`, `[gs]`, `[oss]`, `[ssh]`. To include all optional dependencies, use `[all]`.
- Run this command to setup google cloud bucket as your storage `dvc remote add -d myremote gs://<mybucket>`
- In order for DVC to be able to push and pull data from the remote, you need to have valid GCP credentials.
- Run the following command for authentication `dvc remote modify --lab2 credentialpath <YOUR JSON TOKEN LOCATION>`
### Tracking Data with DVC
- Ensure you have downloaded the [required data](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) and placed it in the "data" folder, renaming the file to "CC_GENERAL.csv."
- To initiate data tracking, execute the following steps:
	1. Run the `dvc init` command to initialize DVC for your project. This will generate a `.dvc` file that stores metadata and configuration details. Your `.dvc` file config metadata will look something like this
	```
    [core]
        remote = lab2
    ['remote "lab2"']
        url = gs://ie7374
	```

	dvc remote modify --git-action-gcp credentialpath git-action-gcp git-action-gcp-3e47dbac54ff.json 
	2. Next, use `dvc add data/CC_GENERAL.csv` to instruct DVC to start tracking this specific dataset.
	3. To ensure version control, add the generated `.dvc` file to your Git repository with `git add data/CC_GENERAL_csv.dvc`.
	4. Also, include the `.gitignore` file located in the "data" folder in your Git repository by running `git add data/.gitignore`.
	5. To complete the process, commit these changes with Git to record the dataset tracking configuration.

- To push your data to the remote storage in Google Cloud, use the following DVC command: `dvc push` This command will upload your data to the Google Cloud Storage bucket specified in your DVC configuration, making it accessible and versioned in the cloud.

### Handling Data Changes and Hash Updates
Whenever your dataset undergoes changes, DVC will automatically compute a new hash for the updated file. Here's how the process works:
- **Update the Dataset:** Replace the existing "CC_GENERAL.csv" file in the "data" folder with the updated version.
- **Update DVC Tracking:** Execute `dvc add data/CC_GENERAL.csv` again to update DVC with the new version of the dataset. When DVC computes the hash for the updated file, it will be different from the previous hash, reflecting the changes in the dataset.
- **Commit and Push:** Commit the changes with Git and push them to your Git repository. This records the update to the dataset, including the new hash.
- **Storage in Google Cloud:** When you run dvc push, DVC uploads the updated dataset to the Google Cloud Storage bucket specified in your DVC configuration. Each version of the dataset is stored as a distinct object within the bucket, organized for easy retrieval.
#### Reverting to Previous Versions with Hashes
To revert to a previous dataset version:
- **Checkout Git Commit:** Use Git to checkout the specific commit where the desired dataset version was last committed. For example, run `git checkout <commit-hash>`
- **Use DVC:** After checking out the Git commit, use DVC to retrieve the dataset version corresponding to that commit by running `dvc checkout`. DVC uses the stored hash to identify and fetch the correct dataset version associated with that commit.

> 💡You can follow [this](https://www.youtube.com/watch?v=kLKBcPonMYw&list=PL7WG7YrwYcnDb0qdPl9-KEStsL-3oaEjg&pp=iAQB) tutorial to learn about DVC in detail.
---

## Lab Extension: Versioned Data → Reproducible ML Pipeline

The original lab only versions a toy `data.txt`. This extension turns the folder into a full DVC project that versions a **real dataset**, runs a **multi-stage ML pipeline** (`dvc.yaml`), tracks **params, metrics and plots**, and shows how a change in data version changes the model.

### Dataset
[UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) — physicochemical measurements of Portuguese "Vinho Verde" wines with a 0–10 quality score. The task is binary classification: **is the wine "good" (quality ≥ 7)?**

Two dataset versions are tracked with `dvc add` to exercise data versioning on a real change:

| version | contents | rows | `data/raw/wine.csv` md5 |
|---|---|---|---|
| v1 | red wines only | 1,599 | `75596e6c9e84942837fd8341b089210e` |
| v2 | red + white wines | 6,497 | `ea2baa6730c01e8502c72548c933388a` |

`src/get_data.py --version v1|v2` rebuilds either version straight from UCI, so the data is reproducible even without access to the remote.

### Project layout
```
Lab_1/
├── data/raw/wine.csv.dvc     # DVC pointer to the versioned raw dataset
├── dvc.yaml                  # pipeline: prepare -> train -> evaluate
├── dvc.lock                  # exact hashes of every dep/out of the last run
├── params.yaml               # all tunable knobs (tracked by DVC)
├── src/
│   ├── get_data.py           # download + build dataset v1 / v2
│   ├── prepare.py            # clean, dedupe, feature engineering, split, winsorize
│   ├── train.py              # RandomForest | GradientBoosting | LogisticRegression
│   ├── evaluate.py           # metrics + ROC / PR / confusion / feature-importance data
│   └── compare_runs.py       # builds reports/version_comparison.md
├── metrics/                  # data_report.json, scores.json (DVC metrics)
├── plots/                    # CSVs rendered by `dvc plots show`
└── reports/                  # snapshots of each run + comparison table
```

### Pipeline stages
1. **prepare** — drops NA rows and exact duplicates (240 in v1, 1,177 in v2), engineers `is_red`, `free_so2_ratio`, `acidity_total`, `alcohol_x_sulphates`, makes a stratified 80/20 split, and winsorizes outliers using quantiles computed **on the train split only** (no test leakage). Writes `metrics/data_report.json`.
2. **train** — fits the model named in `params.yaml → train.model`.
3. **evaluate** — scores the held-out split (accuracy, precision, recall, F1, ROC-AUC, average precision) into `metrics/scores.json` and writes plot data (confusion matrix, ROC, PR curve, permutation feature importance).

DVC only re-runs stages whose deps/params changed, and the run-cache restores outputs instantly when a previous configuration is repeated.

### Running it
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 src/get_data.py --version v2   # or v1
dvc add data/raw/wine.csv             # version the raw data
dvc repro                             # run the pipeline
dvc metrics show                      # view metrics
dvc plots show                        # renders dvc_plots/index.html
dvc push                              # upload data + model to the default remote
```

Try a different model:
```bash
# option A: edit params.yaml (train.model: gradient_boosting), then
dvc repro
dvc params diff && dvc metrics diff           # compare against the last commit

# option B: DVC experiments (needs the pipeline committed to git first)
dvc exp run -S train.model=gradient_boosting
dvc exp show
```

### Remotes
- `localstore` (default) — `Lab_1/dvc_storage/`, a local folder remote so `dvc push`/`dvc pull` work with no cloud setup (git-ignored).
- `myremote` — the GCS bucket from the original lab. Use it with
  `dvc remote modify --local myremote credentialpath <path-to-key.json>` and `dvc push -r myremote`.
  The `--local` flag keeps the key path in `.dvc/config.local`, which is git-ignored, so credentials never end up in git.

### Rolling back the data version
```bash
git checkout <commit-with-v1> -- data/raw/wine.csv.dvc
dvc checkout data/raw/wine.csv.dvc     # restores the 1,599-row v1 file from cache
dvc repro                              # pipeline re-runs (or is restored from run-cache)
```
This was verified: swapping the pointer to the v1 hash and running `dvc checkout` restored the v1 file, and switching back restored v2, without re-downloading anything.

### Results

| data | model | accuracy | precision | recall | F1 | ROC-AUC | avg precision |
|---|---|---|---|---|---|---|---|
| v1 | random_forest | 0.8640 | 0.5000 | 0.6216 | 0.5542 | 0.8922 | 0.6213 |
| v2 | gradient_boosting | 0.8346 | 0.6066 | 0.3663 | 0.4568 | 0.8496 | 0.5704 |
| v2 | logistic_regression | 0.7293 | 0.3925 | 0.7772 | 0.5216 | 0.8154 | 0.4640 |
| v2 | **random_forest** | 0.8130 | 0.5051 | 0.7426 | **0.6012** | 0.8695 | 0.6202 |

Observations:
- Going from v1 → v2 quadrupled the data and raised the positive rate from 13.5% to 19.0%. The random forest's **F1 improved from 0.554 to 0.601** and recall from 0.62 to 0.74, while accuracy and ROC-AUC dipped slightly. White wines are a harder, more varied population, so v1 and v2 test sets are not directly comparable on accuracy.
- On v2 the class-balanced random forest gives the best F1; gradient boosting is the most precise but misses most good wines (recall 0.37); logistic regression has high recall but the lowest precision.
- Permutation importance (ROC-AUC drop) ranks **alcohol** far ahead of every other feature (0.115), followed by volatile acidity, chlorides and the engineered `alcohol_x_sulphates`.

The full table is regenerated with `python3 src/compare_runs.py` → `reports/version_comparison.md`.
