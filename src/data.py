from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Base de Dados"


def _ascii(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()


def _num(series):
    # Os CSVs misturam valores já inferidos como float (ex.: "12.0") e
    # decimais brasileiros em texto (ex.: "7,5"). Não remover pontos evita
    # transformar 12.0 em 120.
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False),
                         errors="coerce")


def _phase_number(value):
    text = _ascii(value).upper().strip()
    if text in {"ALFA", "ALPHA"}:
        return 0.0
    found = re.search(r"\d+", text)
    return float(found.group()) if found else np.nan


def load_year(year, data_dir=DATA_DIR):
    path = Path(data_dir) / f"pede_{year}.csv"
    kwargs = {"sep": "," if year == 2024 else ";"}
    if year != 2024:
        kwargs["encoding"] = "latin1"
    df = pd.read_csv(path, **kwargs)
    df = df[df["RA"].notna()].copy()
    df["ano"] = year
    df["fase_num"] = df["Fase"].map(_phase_number)
    df["idade"] = _num(df["Idade 22"] if year == 2022 else df["Idade"])
    df["anos_programa"] = year - _num(df["Ano ingresso"])
    df["genero"] = df["Gênero"].map(lambda x: _ascii(x).strip().title())
    df["instituicao"] = df["Instituição de ensino"].map(
        lambda x: _ascii(x).strip().title())
    rename = {
        "IAA": "iaa", "IEG": "ieg", "IPS": "ips", "IPP": "ipp",
        "IDA": "ida", "IPV": "ipv", "IAN": "ian",
    }
    for old, new in rename.items():
        df[new] = _num(df[old]) if old in df else np.nan
    df["inde"] = _num(df["INDE 22"] if year == 2022 else df[f"INDE {year}"])
    defas_col = "Defas" if year == 2022 else "Defasagem"
    df["defasagem"] = _num(df[defas_col])
    stone_col = "Pedra 22" if year == 2022 else f"Pedra {year}"
    df["pedra"] = df[stone_col] if stone_col in df else np.nan
    return df


def load_panel(data_dir=DATA_DIR):
    frames = [load_year(year, data_dir) for year in (2022, 2023, 2024)]
    return pd.concat(frames, ignore_index=True), {y: f for y, f in zip((2022, 2023, 2024), frames)}


NUMERIC_FEATURES = [
    "idade", "anos_programa", "fase_num", "iaa", "ieg", "ips",
    "ida", "ipv", "ian", "inde", "defasagem",
]
CATEGORICAL_FEATURES = ["genero", "instituicao"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_transition(current, following):
    target = following[["RA", "defasagem", "ida", "ieg"]].rename(columns={
        "defasagem": "defasagem_seguinte",
        "ida": "ida_seguinte",
        "ieg": "ieg_seguinte",
    })
    out = current.merge(target, on="RA", how="inner")
    out["risco_seguinte"] = (out["defasagem_seguinte"] < 0).astype(int)
    out["queda_ida"] = out["ida_seguinte"] < (out["ida"] - 0.5)
    out["queda_ieg"] = out["ieg_seguinte"] < (out["ieg"] - 0.5)
    return out


def make_transitions(years):
    train = make_transition(years[2022], years[2023])
    test = make_transition(years[2023], years[2024])
    return train, test
