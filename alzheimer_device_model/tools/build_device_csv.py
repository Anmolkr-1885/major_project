"""
Build the device CSV files used by alzheimer_device_model.py directly from the
three Origin projects (.opju).  Origin itself is NOT required.

    python tools/build_device_csv.py \
        --idvg  DEASD_IGZO_ZrO2_MgO_completeIDVGVD.opju \
        --sens  DEASD_Sensitization_complete.opju \
        --epsc  DEASD_ZRo2_1_MgO_1_12_C1.opju \
        --out   device_data  --raw-out origin_sheets_raw

Column/case mapping below was read from the sheet names and column labels
stored inside the Origin files (and cross-checked numerically).
If you change the Origin files, you can also simply export them from Origin
as CSV in the same tidy format (see README).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opju_reader import extract  # noqa: E402

CASES = {  # case id -> (ZrO2 thickness um, MgO thickness um, pretty label)
    "ZrO2_1_MgO_1": (1.0, 1.0, "ZrO2 1 um / MgO 1 um"),
    "ZrO2_1point5_MgO_point5": (1.5, 0.5, "ZrO2 1.5 um / MgO 0.5 um"),
    "ZrO2_point5_MgO_1point5": (0.5, 1.5, "ZrO2 0.5 um / MgO 1.5 um"),
}
PAIRS5 = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"), ("I", "J")]

# --- IDVG_IDVD project: Book1 sheet index -> (kind, case) --------------------
IDVG_SHEETS = {
    1: ("ID_VG_ZrO2_1_MgO_1", "transfer", "ZrO2_1_MgO_1"),
    2: ("ID_VD_ZrO2_1_MgO_1", "output", "ZrO2_1_MgO_1"),
    3: ("ID_VG_ZrO2_half_MgO_1point5", "transfer", "ZrO2_point5_MgO_1point5"),
    4: ("ID_VD_ZrO2_half_MgO_1point5", "output", "ZrO2_point5_MgO_1point5"),
    5: ("ID_VG_ZrO2_1point5_MgO_half", "transfer", "ZrO2_1point5_MgO_point5"),
    6: ("ID_VD_ZrO2_1point5_MgO_half", "output", "ZrO2_1point5_MgO_point5"),
    7: ("Sheet1_linear_IdVg_Vds1V", "linear", None),
}
VDS_LABELS = [1.0, 2.0, 3.0, 4.0, 5.0]            # legend "1V..5V" of transfer graphs
VG_LABELS_OUTPUT = ["1V", "5V", "10V", "15V", "20V"]  # column labels stored in ID_VD sheets
LINEAR_COLS = [("A", "B", "Experimental"), ("C", "D", "ZrO2_1_MgO_1"),
               ("E", "F", "ZrO2_point5_MgO_1point5"), ("G", "H", "ZrO2_1point5_MgO_point5")]

# --- Sensitization summary project: Book1 sheet -> (condition, variable) -----
SENS_SHEETS = {
    1: ("1equal2_deltvary", "equal", "interval_ms"),
    2: ("1equal2_pwvary1", "equal", "pulse_width_ms"),
    3: ("1great2_delvary", "first_greater", "interval_ms"),
    4: ("1great2_pwvary", "first_greater", "pulse_width_ms"),
    5: ("1great2_magvary", "first_greater", "gate_voltage_V"),
    6: ("1less2_magvary", "first_less", "gate_voltage_V"),
    7: ("1less2_deltvary", "first_less", "interval_ms"),
    8: ("1less2_pwvary", "first_less", "pulse_width_ms"),
}
SENS_COLS = [("A", "B", "ZrO2_1_MgO_1"), ("C", "D", "ZrO2_1point5_MgO_point5"),
             ("E", "F", "ZrO2_point5_MgO_1point5")]

# --- raw EPSC project (case ZrO2 1 / MgO 1): (book, sheet) -> layout ---------
# traces: list of (time col, current col); summary: x, A1, A2, SI columns
EPSC_SHEETS = {
    ("A1equal2", 1): ("deltvary", "equal", "interval_ms", 10, ("U", "V", "W", "X")),
    ("A1equal2", 2): ("pwvary", "equal", "pulse_width_ms", 10, ("U", "V", "W", "X")),
    ("A1great2", 1): ("deltvary", "first_greater", "interval_ms", 10, ("U", "V", "W", "X")),
    ("A1great2", 2): ("pwvary", "first_greater", "pulse_width_ms", 8, ("Q", "R", "S", "T")),
    ("A1great2", 3): ("magvary", "first_greater", "gate_voltage_V", 5, ("K", "L", "M", "N")),
    ("A1less2", 1): ("Magvary", "first_less", "gate_voltage_V", 6, ("M", "N", "O", "P")),
    ("A1less2", 2): ("pwvary", "first_less", "pulse_width_ms", 9, ("S", "T", "U", "V")),
    ("A1less2", 3): ("deltvary", "first_less", "interval_ms", 10, ("U", "V", "W", "X")),
}
# Sheet 9 of the summary project = raw EPSC of the 1st=2nd interval run that the
# summary actually uses (A1 = 6.18 nA, interval 2-20 ms).
SENS_RAW_SHEET = (9, "equal", "interval_ms", 10, ("U", "V", "W", "X"))


def letters(n):
    out = []
    for i in range(n):
        s, k = "", i + 1
        while k:
            k, r = divmod(k - 1, 26)
            s = chr(65 + r) + s
        out.append(s)
    return out


def pair(df, xc, yc):
    x = df[xc].to_numpy(float)
    y = df[yc].to_numpy(float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    return x[ok], y[ok]


def col(df, c):
    v = df[c].to_numpy(float)
    return v[~np.isnan(v)]


def case_meta(case):
    z, m, lab = CASES[case]
    return {"case": case, "zro2_um": z, "mgo_um": m}


def build(idvg, sens, epsc, out, raw_out):
    out.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    s_idvg, _ = extract(idvg)
    s_sens, _ = extract(sens)
    s_epsc, _ = extract(epsc)

    # ---------- raw sheet dumps (wide, one CSV per Origin sheet) ----------
    # (raw EPSC traces are written in tidy form to epsc_traces_*.csv instead)
    for tag, sheets, names in [
        ("IDVG_IDVD", s_idvg, {k: v[0] for k, v in IDVG_SHEETS.items()}),
        ("Sensitization", s_sens, {**{k: v[0] for k, v in SENS_SHEETS.items()}, 9: "Sheet1_raw_EPSC_equal_interval"}),
    ]:
        for (book, idx), df in sheets.items():
            nm = names.get(idx, f"sheet{idx}")
            df.dropna(how="all").to_csv(raw_out / f"{tag}__{nm}.csv", index=False)

    # ---------- transfer & output curves ----------
    tr_rows, out_rows = [], []
    for idx, (sheet, kind, case) in IDVG_SHEETS.items():
        df = s_idvg[("Book1", idx)]
        if kind == "transfer":
            for vds, (xc, yc) in zip(VDS_LABELS, PAIRS5):
                vg, i_d = pair(df, xc, yc)
                tr_rows.append(pd.DataFrame({**case_meta(case), "Vds_V": vds, "Vg_V": vg, "Id_A": i_d}))
        elif kind == "output":
            for lab, (xc, yc) in zip(VG_LABELS_OUTPUT, PAIRS5):
                vd, i_d = pair(df, xc, yc)
                out_rows.append(pd.DataFrame({**case_meta(case), "Vg_label": lab, "Vd_V": vd, "Id_A": i_d}))
    transfer = pd.concat(tr_rows, ignore_index=True)
    transfer.to_csv(out / "transfer_curves.csv", index=False, float_format="%.6e")
    pd.concat(out_rows, ignore_index=True).to_csv(out / "output_curves.csv", index=False, float_format="%.6e")

    # ---------- linear Id-Vg (Vds = 1 V) incl. experimental ----------
    df = s_idvg[("Book1", 7)]
    lin = []
    for xc, yc, name in LINEAR_COLS:
        vg, i_d = pair(df, xc, yc)
        if name != "Experimental":  # sanity check against the transfer sheet
            t = transfer[(transfer.case == name) & (transfer.Vds_V == 1.0)]
            ref = np.interp(vg, t.Vg_V, t.Id_A)
            assert np.allclose(ref, i_d, rtol=1e-9), f"linear-plot column {yc} != {name}"
        lin.append(pd.DataFrame({"curve": name, "Vds_V": 1.0, "Vg_V": vg, "Id_A": i_d}))
    pd.concat(lin, ignore_index=True).to_csv(out / "linear_IdVg_Vds1V.csv", index=False, float_format="%.6e")

    # ---------- sensitization index (all 3 cases) ----------
    si_rows = []
    for idx, (sheet, cond, var) in SENS_SHEETS.items():
        df = s_sens[("Book1", idx)]
        for xc, yc, case in SENS_COLS:
            x, si = pair(df, xc, yc)
            si_rows.append(pd.DataFrame({**case_meta(case), "condition": cond, "variable": var,
                                         "x": x, "SI": si, "origin_sheet": sheet}))
    sens_df = pd.concat(si_rows, ignore_index=True)
    sens_df.to_csv(out / "sensitization_index.csv", index=False, float_format="%.6g")

    # ---------- raw EPSC traces + peaks (case ZrO2 1 / MgO 1 only) ----------
    tr_list, pk_list = [], []

    def add_epsc(df, cond, var, ntr, summ, source):
        xs, a1c, a2c, sic = summ
        xv = col(df, xs)
        a2 = col(df, a2c)
        si = col(df, sic)
        a1 = col(df, a1c) if a1c in df else a2 / si
        pk_list.append(pd.DataFrame({"case": "ZrO2_1_MgO_1", "condition": cond, "variable": var, "x": xv,
                                     "A1_nA": a1[: len(xv)], "A2_nA": a2, "SI": si, "source": source}))
        L = letters(2 * ntr)
        for k in range(ntr):
            t, i = pair(df, L[2 * k], L[2 * k + 1])
            tr_list.append(pd.DataFrame({"case": "ZrO2_1_MgO_1", "condition": cond, "variable": var,
                                         "x": xv[k], "time_s": t, "current_A": i, "source": source}))

    idx, cond, var, ntr, summ = SENS_RAW_SHEET
    add_epsc(s_sens[("Book1", idx)], cond, var, ntr, summ, "Sensitization_complete:Sheet1")
    for key, (sheet, cond, var, ntr, summ) in EPSC_SHEETS.items():
        src = f"EPSC_file:{key[0]}/{sheet}"
        add_epsc(s_epsc[key], cond, var, ntr, summ, src)
    pd.concat(tr_list, ignore_index=True).to_csv(out / "epsc_traces_ZrO2_1_MgO_1.csv", index=False,
                                                 float_format="%.6g")
    pd.concat(pk_list, ignore_index=True).to_csv(out / "epsc_peaks_ZrO2_1_MgO_1.csv", index=False,
                                                 float_format="%.6g")

    # ---------- device figures of merit ----------
    summ_rows = []
    for case in CASES:
        t = transfer[(transfer.case == case) & (transfer.Vds_V == 1.0)].sort_values("Vg_V")
        vg, i_d = t.Vg_V.to_numpy(), t.Id_A.to_numpy()
        lg = np.log10(np.abs(i_d))
        ss_point = 1000.0 / np.max(np.gradient(lg, vg))                # mV/dec
        dv = [vg[i + np.argmax(lg[i:] >= lg[i] + 1)] - vg[i]
              for i in range(len(vg)) if np.any(lg[i:] >= lg[i] + 1)]
        ss_decade = 1000.0 * min(dv)                                   # mV for a 10x rise
        gm = np.gradient(i_d, vg)
        k = int(np.argmax(gm))
        vth = vg[k] - i_d[k] / gm[k]                                   # ELR intercept
        w = (vg >= -1.0 - 1e-9) & (vg <= 3.0 + 1e-9)
        s = sens_df[sens_df.case == case]
        summ_rows.append({
            **case_meta(case),
            "SS_min_pointwise_mV_per_dec": round(ss_point, 1),
            "min_dVg_for_10x_Id_mV": round(ss_decade, 1),
            "Vth_ELR_intercept_V_at_Vds1V": round(vth, 3),
            "Ion_A_at_Vg5V_Vds1V": i_d[-1],
            "Ioff_min_A_Vds1V": i_d.min(),
            "on_off_ratio_Vds1V": i_d.max() / i_d.min(),
            "Gmin_S_window_m1_to_3V": i_d[w].min() / 1.0,
            "Gmax_S_window_m1_to_3V": i_d[w].max() / 1.0,
            "n_sensitization_curves": s.groupby(["condition", "variable"]).ngroups,
            "SI_min": s.SI.min(), "SI_max": s.SI.max(),
        })
    pd.DataFrame(summ_rows).to_csv(out / "device_summary.csv", index=False, float_format="%.6g")
    print("written:", sorted(p.name for p in out.glob("*.csv")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idvg", required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--epsc", required=True)
    ap.add_argument("--out", default="device_data")
    ap.add_argument("--raw-out", default="origin_sheets_raw")
    a = ap.parse_args()
    build(a.idvg, a.sens, a.epsc, Path(a.out), Path(a.raw_out))
