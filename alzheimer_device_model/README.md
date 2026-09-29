# Alzheimer's prediction with IGZO / ZrO₂–MgO synaptic transistor data

Is folder mein do cheezein hain:

1. Tumhare teeno Origin projects (`.opju`) ka **poora device data CSV mein**. Isse nikalne ke liye Origin ki zaroorat nahi padi.
2. **Device-aware ML model** (`alzheimer_device_model.py`). Ye model Alzheimer's prediction ke liye ek patient CSV leta hai, aur har ZrO₂/MgO stack ke measured sensitization response aur transfer curve ko neural network ke andar use karta hai.

---

## 1. Kitne models? → 3 (har ZrO₂/MgO stack ka ek model)

| Model (auto-created) | Device | Model mein kya use hota hai |
|---|---|---|
| `ZrO2_1_MgO_1` | ZrO₂ 1 µm / MgO 1 µm | us device ke 8 sensitization curves + us device ka transfer curve |
| `ZrO2_1point5_MgO_point5` | ZrO₂ 1.5 µm / MgO 0.5 µm | same |
| `ZrO2_point5_MgO_1point5` | ZrO₂ 0.5 µm / MgO 1.5 µm | same |
| `software_logreg`, `software_mlp` | koi device nahi | sirf reference ke liye, same CV folds pe |

**9 ya 24 models kyun nahi?**
- Har composition ek alag physical device hai, isliye comparison bhi har stack ke hisaab se hona chahiye ("kaunsa stack Alzheimer's prediction ke liye best hai?").
- 3 stimulus conditions (1st = 2nd, 1st > 2nd, 1st < 2nd) × 3 variables (interval, pulse width, gate voltage) = **8 measured curves** banti hain (1st = 2nd ka gate-voltage sweep measure nahi hua tha). Ye alag-alag models nahi hain. Ye **ek hi device ke 8 "virtual nodes"** hain: har patient feature 8 alag nonlinear device responses mein convert hota hai. Agar har curve ka alag model banate, toh har model device ka sirf ek hissa dekhta.
- `--ablation` flag har condition ko akele chala kar dikhata hai ki kaunsi condition kitni information carry karti hai. Isme 9 extra runs hote hain, isi script se, aur `fig7` ban jaata hai.
- Nayi composition add karni ho toh CSV mein rows add kar do. Script har unique `case` ke liye apne aap ek model bana deta hai.

---

## 2. Model kaise kaam karta hai (physical reservoir computing)

```
patient CSV  ──clean (impute, one-hot, 0–1 scaling)──►  x₁ … x_d
   │  har feature x ek paired-pulse stimulus ka parameter set karta hai
   │  (interval 2–20 ms | pulse width 1–10 ms | amplitude 1.0–1.5 V), teeno conditions mein
   ▼
device response = measured Sensitization Index  SI = A2/A1   (8 curves, PCHIP interpolation)
   ▼
8·d node states  (ye layer TRAIN NAHI hoti – ye device physics hai)
   ▼
linear readout (logistic regression, sirf yahi layer train hoti hai)
   ▼
readout weights device conductance mein store hote hain:
G = Id/Vds transfer curve se (Vds = 1 V, Vg −1…3 V, 32 levels), differential pair G⁺ − G⁻
   ▼
P(Alzheimer's)
```

`[float w]` = ideal software weights, `[device w]` = device levels mein quantized weights (optionally noise ke saath).
**Paper mein `[device w]` wala result report karna chahiye.** Dono ka difference dikhata hai ki hardware ki wajah se kitni accuracy loss hoti hai.

---

## 3. Run kaise karein

```bash
pip install -r requirements.txt

# 1) jaldi test (synthetic demo data – sirf check ke liye)
python alzheimer_device_model.py train --data demo_patients_SYNTHETIC.csv --target Diagnosis --positive Demented

# 2) apna data
python alzheimer_device_model.py train --data your_patients.csv --target Diagnosis --ablation

# 3) naye patients ke liye prediction
python alzheimer_device_model.py predict --model results/models/ZrO2_1_MgO_1.joblib \
       --data new_patients.csv --out predictions.csv
```

Public datasets ke liye ready commands:

| Dataset | Command options |
|---|---|
| Kaggle "Alzheimer's Disease Dataset" (2149 patients) | `--target Diagnosis` |
| OASIS-1 cross-sectional | `--target CDR --positive 0.5,1,2` (CDR > 0 = dementia) |
| OASIS-2 longitudinal | `--target Group --positive Demented,Converted --drop CDR,Visit,"MR Delay" --group-col "Subject ID"` |

`--group-col` se ek subject ki saari visits ek hi fold mein rehti hain. Isse leakage nahi hota.

Saved model ko apne notebook mein load karna ho:
```python
import alzheimer_device_model, joblib      # file same folder mein honi chahiye
bundle = joblib.load("results/models/ZrO2_1_MgO_1.joblib")
```

---

## 4. Patient CSV ka format

- Ek row = ek patient (ya ek visit). Columns kuch bhi ho sakte hain. Label column `--target` se batao.
- Numeric aur text dono tarah ke columns (M/F, Yes/No) chalte hain. Missing values median ya most-frequent value se bhar di jaati hain.
- ID columns, constant columns aur 50% se zyada missing wale columns apne aap hat jaate hain (console pe dikhta hai).
- **Leakage check:** agar koi column akele hi label predict kar deta hai (jaise OASIS mein `CDR` se `Group` banta hai), toh script warning deta hai. Us column ko `--drop CDR` se hata do.

---

## 5. Device CSVs (`device_data/`)

| File | Model use karta hai? | Columns |
|---|---|---|
| `sensitization_index.csv` | ✅ reservoir nodes | case, zro2_um, mgo_um, condition, variable, x, SI, origin_sheet |
| `transfer_curves.csv` | ✅ synaptic weights | case, zro2_um, mgo_um, Vds_V, Vg_V, Id_A |
| `output_curves.csv` | reference | case, Vg_label, Vd_V, Id_A |
| `linear_IdVg_Vds1V.csv` | reference | curve (Experimental + 3 cases), Vg_V, Id_A |
| `epsc_peaks_ZrO2_1_MgO_1.csv` | reference | condition, variable, x, A1_nA, A2_nA, SI, source |
| `epsc_traces_ZrO2_1_MgO_1.csv` | reference | condition, variable, x, time_s, current_A, source |
| `device_summary.csv` | reference | SS, Vth, on/off ratio, Gmin/Gmax (data se calculate kiya) |

`condition`: `equal` (1st = 2nd), `first_greater` (1st > 2nd), `first_less` (1st < 2nd).
`variable`: `interval_ms`, `pulse_width_ms`, `gate_voltage_V`.

- `origin_sheets_raw/` mein har Origin sheet ek wide CSV ke roop mein hai (Origin wale column letters ke saath).
- `tools/build_device_csv.py` se ye saari CSVs Origin files se dobara ban jaati hain. Agar Origin data mein kuch theek karo toh ise dobara chala lo.
- Chaho toh Origin se khud bhi export kar sakte ho, bas upar wale column format mein.

---

## 6. Outputs (`results/`)

| File | Kya hai |
|---|---|
| `report.md` | paper-ready table (mean ± sd), noise / levels / ablation tables |
| `metrics_summary.csv`, `metrics_per_fold.csv` | accuracy, balanced accuracy, sensitivity, specificity, F1, ROC-AUC |
| `robustness_noise.csv`, `robustness_levels.csv` | accuracy vs conductance variation σ, aur accuracy vs number of levels |
| `ablation_conditions.csv` | kaunsi stimulus condition kitni useful hai (`--ablation` ke saath) |
| `oof_predictions.csv`, `device_levels.csv`, `run_config.json` | predictions, programmed conductance levels, run settings |
| `models/*.joblib` | saare patients pe train kiye gaye final models (`predict` ke liye) |
| `figures/fig1…fig7` | model comparison, ROC, device nodes, conductance levels, robustness, confusion matrices, ablation |

---

## 7. Useful options

| Option | Default | Matlab |
|---|---|---|
| `--n-levels` | 32 | har synapse device ke conductance states |
| `--vg-min / --vg-max` | −1 / 3 V | weight programming ka gate window (tumhare linear Id–Vg plot jaisa) |
| `--read-vds` | 1 V | kis Vds wale transfer curve se weights banenge |
| `--noise` | 0 | main result mein relative conductance variation (0.05 = 5 %) |
| `--repeats` | 1 | chhote datasets (<500 patients) ke liye 5 rakho |
| `--skip-mlp` | – | MLP reference skip karta hai, run bahut fast ho jaata hai |
| `--ablation` | – | har condition akele (3 extra runs per case) |

**Tip:** −1…3 V window mein lagbhag aadhe levels subthreshold region mein aate hain, jahan conductance ≈ 0 hai, toh wo waste ho jaate hain (fig4 dekho). `--vg-min 1.0` (sirf above-threshold programming) se Kaggle test pe device-weight balanced accuracy 0.873 se 0.880 ho gayi, aur 16 levels pe 0.78 se 0.86.

---

## 8. Data mein jo issues mile (paper se pehle zaroor check karo)

1. **SS 10× chhota likha hai.** Graph pe likha hai "SS = δVg/δIds = 560 mV / 10 = 56 mV/dec", lekin 10× current change ek decade hota hai, toh SS = ΔVg/Δlog₁₀(Id) = **560 mV/dec** banta hai. Data se bhi yahi nikalta hai: ek decade ke liye kam se kam 525 mV (ZrO₂ 1.5/MgO 0.5), 625 mV (1/1) aur 750 mV (0.5/1.5) lagte hain (`device_summary.csv`). Waise bhi room temperature pe SS 60 mV/dec se neeche nahi ja sakta.
2. **`1less2_magvary`:** ZrO₂ 1.5/0.5 aur ZrO₂ 0.5/1.5 ke columns bilkul same hain (13.85 → 19.48). Lagta hai copy-paste ho gaya.
3. **`ID_VD_ZrO2_1_MgO_1`:** "1V" label wale column ka current "20V" label wale column se zyada hai (baaki do cases mein order sahi hai). Shayad labels ek position shift ho gaye hain. CSV mein labels waise hi rakhe hain jaise file mein the.
4. **Linear Id–Vg plot:** data ke hisaab se E–F column ZrO₂ 0.5/MgO 1.5 hai aur G–H column ZrO₂ 1.5/MgO 0.5. Legend ka order check kar lena.
5. **EPSC file ka `A1equal2/deltvary`** (1–10 ms, A1 = 3.57 nA, SI 9.97 → 2.11) summary ke `1equal2_deltvary` se alag run hai. Summary wala data Sensitization file ke Sheet1 se aata hai (2–20 ms, A1 = 6.18 nA). Dono runs `epsc_peaks_*.csv` mein `source` column ke saath rakhe hain.

---

## 9. Assumptions (paper ke Methods section mein likhna)

- **Input encoding:** har feature (0–1) ko har curve ke measured parameter window pe linearly map kiya gaya hai.
- **Node output:** SI = A2/A1. Measured points (5–10 per curve) ke beech shape-preserving PCHIP interpolation use hua hai.
- **Synapse:** Vds = 1 V pe gate-programmed channel conductance. Levels Vg mein evenly spaced hain (DAC steps). Har weight ek differential pair hai. Bias aur har node ka gain/offset peripheral circuit mein hai.
- **Noise:** har device pe multiplicative Gaussian σ. Ye assumption hai, kyunki curves mein cycle-to-cycle spread nahi hai.
- Weights post-training quantize kiye gaye hain (device pe retraining nahi hui).

---

## 10. Pipeline check on public data (final result nahi hai, sirf ye dikhane ke liye ki code chalta hai)

Balanced accuracy / ROC-AUC, device weights (32 levels, −1…3 V, noise 0):

| Model | Kaggle AD dataset (2149, 5-fold) | OASIS-1, CDR>0 (235, 5-fold × 3) |
|---|---|---|
| Software logistic regression | 0.817 / 0.898 | 0.811 / 0.905 |
| Software MLP | 0.823 / 0.908 | 0.807 / 0.906 |
| ZrO₂ 1 / MgO 1 | **0.873 / 0.926** | **0.807 / 0.902** |
| ZrO₂ 1.5 / MgO 0.5 | 0.859 / 0.920 | 0.782 / 0.886 |
| ZrO₂ 0.5 / MgO 1.5 | 0.853 / 0.920 | 0.789 / 0.895 |

`demo_patients_SYNTHETIC.csv` random fake data hai, sirf test ke liye (`tools/make_demo_data.py`). Iske results kahin report mat karna.

Real data ke sources: [OASIS](https://sites.wustl.edu/oasisbrains/), [Kaggle Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset), [ADNI](https://adni.loni.usc.edu) (access ke liye application dena padta hai). Jo bhi dataset use karo, uske data-use terms follow karo aur paper mein cite karo.
