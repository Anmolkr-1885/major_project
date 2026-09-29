# Alzheimer's MRI classification: CNN chosen from the 4 reference papers

Yeh folder Mendeley dataset ([ch87yswbz4](https://data.mendeley.com/datasets/ch87yswbz4/1)) ke brain-MRI slices se Alzheimer's stage predict karne ka code hai. Ek hi file hai: `alzheimer_mri_cnn.py`.

---

## 1. Papers se model kaise chuna

| Paper | Paper kya kehta hai | Hamare kaam ke liye |
|---|---|---|
| Ref 1: Galal et al., *Neuroscience* 2025 (review) | Imaging (MRI) data ke liye supervised classification mein CNN / deep NN / SVM; SVM aur RF sabse common | Direction mila: MRI → CNN, aur SVM baseline |
| Ref 2: Kate et al., *CNSE* 2024 (ANN review) | MRI → resize/normalize → CNN feature extraction → kernel SVM; metrics: accuracy, precision, recall, F1, AUC; VGG16 hippocampus analysis | CNN + SVM baseline, wahi metrics, VGG16 optional |
| Ref 3: Azeem et al., arXiv 2021 | Images ke liye CNN sabse suitable; **ConXNet** = 4 × (Conv → ReLU → BatchNorm → MaxPool) → Dense, Keras, Adam lr 0.001, batch 32; Alzheimer's CNNs 93–98.8% (4-class bhi); augmentation, dropout, heat-maps | ✅ **Chuna gaya model** |
| Ref 4: Zhang et al., *IEEE TMI* 2026 (UniTransAD) | Bina labels ka anomaly detection (tumor, stroke, MS lesions); healthy multi-sequence 3D MRI pe train; output lesion map hota hai | ❌ Fit nahi: hamare paas labelled 2D single-sequence slices hain; Alzheimer's ek diffuse atrophy hai, focal lesion nahi; aur yeh model class predict nahi karta |

**Final model:** ConXNet-style CNN (Ref 3) jisme 4 classes ka softmax output hai. Isme Refs 2–3 ke suggestions bhi daale gaye hain: augmentation, dropout, class weights aur Grad-CAM heat-maps.

**Ek badlav:** Flatten ki jagah Global Average Pooling use kiya hai, taaki parameters kam rahein aur chhote dataset pe overfitting kam ho.

Isi split pe 3 reference baselines bhi train hote hain:
- ANN/MLP (Refs 2–3)
- PCA + SVM (Ref 1)
- CNN features + kernel SVM (Ref 2)

VGG16 transfer learning (Ref 2) optional hai (`--arch vgg16`). Iske liye ImageNet weights download karne padte hain, isliye internet chahiye.

---

## 2. Dataset mein kya mila (zaroor padhna)

- Zip mein do folders hain: `OriginalDataset` (6,400 JPG, 176×208) aur `AugmentedAlzheimerDataset` (33,984).
- **6,400 images sirf 200 patients ki hain, har patient ke 32 slices.** Patients per class: NonDemented 100, VeryMild 70, Mild 28, Moderate **2**. Yeh grouping slices ki similarity se nikali gayi: 200 groups bane, har group mein exactly 32 slices, aur koi group mixed-class nahi.
- **Random split = leakage:** har image ki sabse milti-julti image 100% cases mein usi class ki hai. Random split se ek hi patient ke slices train aur test dono mein chale jaate hain, aur ~99% accuracy "free" mein mil jaati hai.
- **`AugmentedAlzheimerDataset`** usi 200 patients ki rotated/zoomed copies hai. Code is folder ko apne aap chhod deta hai; training ke waqt augmentation on-the-fly hota hai.
- **Isliye default = patient-level split:** 130 patients train, 29 val, 41 test, aur train-test mein 0 common patients. Moderate ke sirf 2 patients hain (1 train, 1 test), toh uska result sirf indicative hai. `--merge_moderate` se Mild+Moderate ko ek class bana sakte ho.

---

## 3. Run kaise karein

```bash
pip install -r requirements.txt

# zip unzip karo, phir us folder ka path do (Augmented folder apne aap skip hoga)
python alzheimer_mri_cnn.py inspect --data_dir "Alzheimer_s dataset"      # patients / leakage check (5 sec)

python alzheimer_mri_cnn.py train --data_dir "Alzheimer_s dataset"        # ASLI result: patient-level split
python alzheimer_mri_cnn.py train --data_dir "Alzheimer_s dataset" --split random --out results_random   # sirf leakage dikhane ke liye
python alzheimer_mri_cnn.py train --data_dir "Alzheimer_s dataset" --cv 5 --out results_cv            # 5-fold patient-level CV (paper ke liye best)

python alzheimer_mri_cnn.py predict --model results/model.keras --images new_mri_folder
```

**Time:** CPU pe (2 cores) har epoch ~50 s lagta hai, toh ek run 30–40 min ka hai. GPU / Google Colab pe har run kuch minute ka hai.

**Device option:** `--device_dir ../alzheimer_device_model/device_data` dene se CNN ki dense layers ke weights IGZO device ke 32 conductance levels pe map hote hain. Teeno ZrO₂/MgO cases ka test accuracy `device_mapping.csv` mein aata hai. Yeh pehle wale device model se jodne ke liye hai.

---

## 4. Outputs (`results/`)

| File | Kya hai |
|---|---|
| `metrics_models.csv`, `metrics.json`, `report.md` | CNN + baselines ke test metrics (slice-level aur patient-level) |
| `classification_report.txt` | per-class precision / recall / F1 |
| `predictions_test.csv`, `split_assignment.csv` | har test image ki prediction; kaunsi image kis split mein thi |
| `model.keras`, `class_names.json`, `history_main.csv` | trained model aur training history |
| `device_mapping.csv` | (optional) IGZO levels pe accuracy |
| `figures/` | dataset overview, training curves, confusion matrix (slices aur patients), ROC, Grad-CAM, model comparison |

**Paper mein kya report karna hai:** patient-level split (ya `--cv 5`) ke numbers. Random split ke numbers sirf yeh dikhane ke liye ki leakage se accuracy kitni badh jaati hai.

---

## 5. Useful options

- `--img_size` (default 128)
- `--epochs` (40)
- `--patience` (10)
- `--batch` (32)
- `--lr` (0.001)
- `--seed` (42)
- `--test_frac` / `--val_frac` (0.2 / 0.15)
- `--merge_moderate`
- `--no_baselines`
- `--arch conxnet | mlp | vgg16`
- `--group_threshold` (0.98): slice similarity ki woh limit jiske upar do slices ek hi patient ke maane jaate hain
