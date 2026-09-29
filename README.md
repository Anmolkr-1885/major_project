# Alzheimer's MRI + IGZO Device Project (Windows Guide)

Bhai, is project mein do **separate workflows** hain:

1. **MRI CNN**: MRI JPG/PNG images se 4 class classification (NonDemented, VeryMildDemented, MildDemented, ModerateDemented).
2. **IGZO device model**: measured ZrO2/MgO transistor CSV data par based device-aware tabular model.

MRI model chalane ke liye patient CSV ya Origin `.opju` files required nahi hain. Device model chalane ke liye MRI images required nahi hain. Dono ko ek hi research claim ke roop mein mix mat karna.

> `data/MRI_DATASET_DEMO/` sirf folder-structure template hai; is ZIP mein tumhare real MRI images include nahi hain. Apne existing `OriginalDataset` folder ka path command mein do, ya images template ke andar copy karo. `AugmentedAlzheimerDataset` ko CNN script skip karta hai; original images use karo.

## 1. Folder structure

```text
Alzheimer_Project/
├── README.md
├── setup_windows.bat
├── inspect_dataset_windows.bat
├── train_cnn_windows.bat
├── alzheimer_mri_cnn/
│   ├── alzheimer_mri_cnn.py
│   ├── README.md
│   └── requirements.txt
├── alzheimer_device_model/
│   ├── alzheimer_device_model.py
│   ├── demo_patients_SYNTHETIC.csv  # fake demo data only
│   ├── requirements.txt
│   ├── README.md
│   ├── device_data/                # included measured device CSVs
│   ├── origin_sheets_raw/          # raw exported CSV references
│   └── tools/
├── data/
│   ├── MRI_DATASET_DEMO/           # EMPTY CLASS-FOLDER TEMPLATE
│   │   ├── OriginalDataset/
│   │   │   ├── NonDemented/
│   │   │   ├── VeryMildDemented/
│   │   │   ├── MildDemented/
│   │   │   └── ModerateDemented/
│   │   └── AugmentedAlzheimerDataset/ # optional; don't train on this
│   ├── patient_csv/                # put real tabular CSV here if using device model
│   └── origin_files/               # optional Origin OPJU files
└── results/                        # generated outputs
```

## 2. Put your existing dataset in place (recommended)

Tumhare screenshot mein dataset `Downloads\Alzheimer's dataset` mein hai, aur project folder bhi `Downloads` ke andar rakho. Structure kuch aisa hoga:

```text
Downloads/
├── Alzheimer's dataset/
│   ├── OriginalDataset/
│   │   ├── VeryMildDemented/  # JPG images
│   │   ├── NonDemented/      # JPG images
│   │   ├── MildDemented/     # JPG images
│   │   └── ModerateDemented/ # JPG images
│   └── AugmentedAlzheimerDataset/ # ignored by script
└── Alzheimer_Project/        # extracted ZIP
```

Folder names class labels ke liye use hote hain. Inhe exactly as-is rakho. `OriginalDataset` ko `Alzheimer_Project/data/MRI_DATASET_DEMO/` mein copy karna zaroori nahi; external folder ka path use kar sakte ho.

## 3. One-time setup (Windows)

1. ZIP extract karo.
2. VS Code mein `Alzheimer_Project` folder open karo.
3. `Terminal > New Terminal` open karo.
4. Python 3.10–3.12 install hona chahiye (Python 3.11 recommended).
5. Terminal mein:

```powershell
py -0p
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r alzheimer_mri_cnn/requirements.txt -r alzheimer_device_model/requirements.txt
```

Agar PowerShell execution-policy error de:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternative: project folder mein `setup_windows.bat` double-click karo. Ye venv create karke dependencies install karega. Agar install fail ho, terminal se commands run karke error dekho.

## 4. MRI CNN run commands

Saare commands project root (`Alzheimer_Project`) se run karo. VS Code terminal PowerShell ho toh ye commands copy-paste karo.

**A. Dataset inspect (training nahi):**

```powershell
python alzheimer_mri_cnn/alzheimer_mri_cnn.py inspect --data_dir "..\Alzheimer's dataset"
```

Ye command tab kaam karegi jab `Alzheimer_Project` aur `Alzheimer's dataset` same `Downloads` folder ke andar sibling folders hon. Agar dataset kahin aur hai, path replace karo, example:

```powershell
python alzheimer_mri_cnn/alzheimer_mri_cnn.py inspect --data_dir "D:\YourFolder\Alzheimer's dataset"
```

**B. Small smoke test (sirf code pipeline check):**

```powershell
python alzheimer_mri_cnn/alzheimer_mri_cnn.py train --data_dir "..\Alzheimer's dataset" --epochs 2 --patience 2 --no_baselines --out results/smoke_test
```

**C. Full CNN training (patient-level split):**

```powershell
python alzheimer_mri_cnn/alzheimer_mri_cnn.py train --data_dir "..\Alzheimer's dataset" --out results/mri_patient_split
```

**D. 5-fold patient-level cross-validation (takes longer):**

```powershell
python alzheimer_mri_cnn/alzheimer_mri_cnn.py train --data_dir "..\Alzheimer's dataset" --cv 5 --out results/mri_cv5
```

**E. Optional CNN + IGZO conductance-mapping simulation:**

```powershell
python alzheimer_mri_cnn/alzheimer_mri_cnn.py train --data_dir "..\Alzheimer's dataset" --device_dir alzheimer_device_model/device_data --out results/mri_igzo_mapping
```

This is software-level conductance mapping, not physical training/programming of a transistor. Device mapping is optional and is not required for a basic MRI CNN run.

If you'd rather run the helper batch files, double-click `inspect_dataset_windows.bat` or `train_cnn_windows.bat`. They assume the dataset is in the sibling `Downloads\Alzheimer's dataset` folder.

## 5. Dataset folder format

The CNN expects images under class folders. It searches recursively, using the class folder name as label, and ignores directories with `augment` in the name.

```text
OriginalDataset/
├── NonDemented/
│   ├── image001.jpg
│   └── ...
├── VeryMildDemented/
├── MildDemented/
└── ModerateDemented/
```

`data/MRI_DATASET_DEMO/` contains empty folders and marker text files only. It is not a usable training dataset until you put actual MRI images inside the four class folders. Do not use synthetic/demo images as research data.

## 6. Output files

CNN results are written to the selected `--out` folder, e.g. `results/mri_patient_split/`. Expected files include model weights, metrics, predictions, split assignments, report and figures. Exact files depend on the run options and successful completion.

## 7. Device-aware tabular model (separate optional workflow)

This workflow takes a patient-level tabular CSV, not MRI image folders. A synthetic CSV is included only to test whether the script runs; never report its performance as a real result.

```powershell
python alzheimer_device_model/alzheimer_device_model.py train --data alzheimer_device_model/demo_patients_SYNTHETIC.csv --target Diagnosis --positive Demented --out results/device_demo
```

For a real approved tabular dataset, replace the CSV path and target column as appropriate. Check `alzheimer_device_model/README.md` for data schema, options, and caveats.

## 8. Important research caution

The MRI dataset described in the included documentation has a strong patient/slice relationship and very few ModerateDemented subjects. Use patient-level split/CV, inspect the grouping report, and interpret per-class metrics carefully. Random image split can cause leakage and inflated results. This code is an educational/research pipeline, not a clinical diagnostic tool.

The included transistor data also has documented measurement/label issues; verify them against the original lab/Origin records before publishing conclusions. See `alzheimer_device_model/README.md`.
