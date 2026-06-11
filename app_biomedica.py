import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import chardet

st.set_page_config(
    page_title="Analizador de equipos de temperatura - Capa Biomédica",
    layout="wide"
)
st.title("Generador de gráficos de temperatura")

# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE PARSING ROBUSTO
# ──────────────────────────────────────────────────────────────────────────────

def detectar_encoding(raw_bytes: bytes) -> str:
    """Detecta el encoding del archivo con chardet; usa latin-1 como respaldo."""
    result = chardet.detect(raw_bytes[:10_000])
    enc = result.get("encoding") or "latin-1"
    # Normalizar nombres problemáticos
    enc = enc.lower().replace("-", "_")
    ALIAS = {"ascii": "utf_8", "windows_1252": "cp1252"}
    return ALIAS.get(enc, enc)

def detectar_separador(texto: str) -> str:
    """Detecta el separador más probable probando los más comunes."""
    candidatos = [",", ";", "\t", "|"]
    primeras = [l for l in texto.splitlines()[:20] if l.strip()]
    conteos = {sep: sum(l.count(sep) for l in primeras) for sep in candidatos}
    mejor = max(conteos, key=conteos.get)
    return mejor if conteos[mejor] > 0 else ","

def encontrar_fila_datos(lineas: list[str], sep: str) -> int:
    """
    Busca la primera fila que parezca contener datos numéricos reales
    (al menos 2 columnas con números/fechas).  Devuelve el índice.
    """
    import re
    patron_num  = re.compile(r"[-+]?\d+[\.,]?\d*")
    patron_fecha = re.compile(
        r"\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
        r"|\d{2}:\d{2}"
    )
    for i, linea in enumerate(lineas):
        campos = linea.split(sep)
        hits = sum(
            1 for c in campos
            if patron_num.search(c) or patron_fecha.search(c)
        )
        if hits >= 2:
            return i
    return 0

def limpiar_numero(serie: pd.Series) -> pd.Series:
    """Convierte columna con posibles '+', ',', espacios a float."""
    return pd.to_numeric(
        serie.astype(str)
             .str.replace(r"[+\s]", "", regex=True)
             .str.replace(",", ".", regex=False),
        errors="coerce"
    )

FORMATOS_FECHA = [
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    # ✨ NUEVO: Añadidos formatos para años de 2 dígitos (ej: 05/20/25)
    "%m/%d/%y %H:%M:%S", "%m/%d/%y %H:%M", 
    "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M",
    "%d/%m/%Y",          "%Y-%m-%d",
]

def parsear_timestamps(serie: pd.Series) -> pd.Series:
    """Intenta múltiples formatos de fecha; usa inferencia como último recurso."""
    for fmt in FORMATOS_FECHA:
        try:
            resultado = pd.to_datetime(serie, format=fmt, errors="raise")
            return resultado
        except Exception:
            continue
    
    # Inferencia automática (compatible con versiones nuevas y antiguas de pandas)
    try:
        return pd.to_datetime(serie, format='mixed', errors="coerce")
    except ValueError:
        return pd.to_datetime(serie, infer_datetime_format=True, errors="coerce")

def corregir_ambiguedad_12h(ts: pd.Series) -> pd.Series:
    """
    Corrige el salto hacia atrás que ocurre cuando el logger usa formato 12h
    sin indicador AM/PM: suma 12h a partir del primer retroceso temporal.
    """
    diffs = ts.diff().dt.total_seconds()
    for idx in ts[diffs < -60].index:          # tolerancia de 1 min
        ts.loc[idx:] = ts.loc[idx:] + pd.Timedelta(hours=12)
    return ts

def es_columna_temperatura(col: pd.Series, nombre: str) -> bool:
    """
    Heurística: columna es de temperatura si es numérica y sus valores
    están en el rango razonable de equipos de temperatura (-100 a 150 °C)
    y no tiene más del 80% de NaN.
    """
    if col.dtype == object:
        col = limpiar_numero(col)
    pct_nan = col.isna().mean()
    if pct_nan > 0.8:
        return False
    vals = col.dropna()
    if len(vals) == 0:
        return False
    # Rango heurístico para equipos de temperatura médica/industrial
    return float(vals.min()) >= -100 and float(vals.max()) <= 150

# ──────────────────────────────────────────────────────────────────────────────
# CARGA Y PARSEO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Cargar archivo (.CSV)", type=["csv"])

if uploaded_file:
    raw_bytes = uploaded_file.read()

    # ── 1. Encoding ──────────────────────────────────────────────────────────
    encoding = detectar_encoding(raw_bytes)
    try:
        texto = raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        texto = raw_bytes.decode("latin-1")
        encoding = "latin-1"
        
    # ✨ NUEVO: Normalizar los saltos de línea (Arregla el error de "\r")
    texto = texto.replace('\r\n', '\n').replace('\r', '\n')

    lineas = texto.splitlines()

    # ── 2. Separador ─────────────────────────────────────────────────────────
    sep = detectar_separador(texto)

    # ── 3. Detectar si tiene cabecera estilo "Temperature Data" (Formato A) ──
    es_formato_a = any("Temperature Data" in l or
                        ("P1" in l and "Time" in l) for l in lineas[:30])

    df = None
    errores_parse = []

    try:
        if es_formato_a:
            # Buscar fila con "P1" y "Time"
            fila_inicio = next(
                (i for i, l in enumerate(lineas) if "P1" in l and "Time" in l),
                0
            )
            df = pd.read_csv(
                io.StringIO(texto),
                skiprows=fila_inicio,
                sep=sep,
                skipinitialspace=True,
                engine="python",
            )
            df.columns = [str(c).strip() for c in df.columns]

            # Limpiar columnas P* y convertir a número
            p_cols = [c for c in df.columns if str(c).startswith("P")]
            for col in p_cols:
                df[col] = limpiar_numero(df[col])

            # Timestamp
            if "Date" in df.columns and "Time" in df.columns:
                ts_raw = df["Date"].astype(str) + " " + df["Time"].astype(str)
                df["Timestamp"] = parsear_timestamps(ts_raw)
            elif "Timestamp" in df.columns:
                df["Timestamp"] = parsear_timestamps(df["Timestamp"].astype(str))
            else:
                raise ValueError("No se encontraron columnas de fecha/hora.")

            df = df.sort_values("Timestamp").reset_index(drop=True)

            # Renombrar P* → Sonda *
            mapeo = {c: f"Sonda {c.replace('P', '')}" for c in p_cols}
            df.rename(columns=mapeo, inplace=True)

        else:
            # ── Formato B: tabla "pura" ───────────────────────────────────────
            fila_datos = encontrar_fila_datos(lineas, sep)

            # Intentar con y sin cabecera
            for tiene_header in (False, True):
                try:
                    df_test = pd.read_csv(
                        io.StringIO(texto),
                        header=0 if tiene_header else None,
                        skiprows=fila_datos if not tiene_header else None,
                        sep=sep,
                        skipinitialspace=True,
                        engine="python",
                        on_bad_lines="skip",
                    )
                    if len(df_test.columns) >= 2:
                        df = df_test
                        break
                except Exception as e:
                    errores_parse.append(str(e))
                    continue

            if df is None:
                raise ValueError(
                    "No se pudo interpretar la estructura del CSV. "
                    f"Detalles: {'; '.join(errores_parse)}"
                )

            df.columns = [str(c) for c in df.columns]

            # ── Detectar columnas fecha/hora vs temperatura ───────────────────
            posibles_dt: list[str] = []
            posibles_temp: list[str] = []

            for col in df.columns:
                muestra = df[col].dropna().head(10).astype(str)
                tiene_fecha = muestra.str.match(
                    r"\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
                ).any()
                tiene_hora = muestra.str.match(r"\d{1,2}:\d{2}").any()

                if tiene_fecha or tiene_hora:
                    posibles_dt.append(col)
                else:
                    col_num = limpiar_numero(df[col])
                    if es_columna_temperatura(col_num, col):
                        posibles_temp.append(col)

            # ── Construir Timestamp ───────────────────────────────────────────
            if len(posibles_dt) == 0:
                raise ValueError(
                    "No se encontró ninguna columna con fechas u horas."
                )
            elif len(posibles_dt) == 1:
                df["Timestamp"] = parsear_timestamps(
                    df[posibles_dt[0]].astype(str)
                )
            else:
                # Combinar las dos primeras columnas datetime (fecha + hora)
                ts_raw = (
                    df[posibles_dt[0]].astype(str).str.strip()
                    + " "
                    + df[posibles_dt[1]].astype(str).str.strip()
                )
                df["Timestamp"] = parsear_timestamps(ts_raw)

            df["Timestamp"] = corregir_ambiguedad_12h(df["Timestamp"])
            df = df.sort_values("Timestamp").reset_index(drop=True)

            # Si no hay columnas de temp identificadas, intentar todas las numéricas
            if not posibles_temp:
                for col in df.columns:
                    if col not in posibles_dt and col != "Timestamp":
                        col_num = limpiar_numero(df[col])
                        if col_num.notna().any():
                            posibles_temp.append(col)

            # Convertir columnas de temperatura y renombrar
            mapeo_sondas: dict[str, str] = {}
            for i, col in enumerate(posibles_temp, start=1):
                df[col] = limpiar_numero(df[col])
                nuevo = f"Sonda {i}" if not str(col).startswith("Sonda") else col
                mapeo_sondas[col] = nuevo
            df.rename(columns=mapeo_sondas, inplace=True)

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
        with st.expander("Detalles técnicos"):
            st.code(texto[:2000])
        st.stop()

    # ── Validar timestamp ────────────────────────────────────────────────────
    n_bad = df["Timestamp"].isna().sum()
    if n_bad == len(df):
        st.error(
            "No se pudo interpretar ningún timestamp. "
            "Revisa que el CSV tenga columnas de fecha y hora legibles."
        )
        with st.expander("Primeras líneas del archivo"):
            st.code("\n".join(lineas[:15]))
        st.stop()
    elif n_bad > 0:
        st.warning(
            f"⚠️ {n_bad} filas tienen timestamps inválidos y serán ignoradas."
        )
        df = df.dropna(subset=["Timestamp"])

    # ──────────────────────────────────────────────────────────────────────────
    # INTERFAZ COMÚN
    # ──────────────────────────────────────────────────────────────────────────
    sondas_candidatas = [c for c in df.columns if "Sonda" in c]
    sondas_activas = [
        s for s in sondas_candidatas
        if df[s].notna().any() and df[s].abs().sum() > 0
    ]

    if not sondas_activas:
        st.error(
            "No se detectaron sondas con lecturas válidas. "
            f"Encoding: {encoding} | Separador: {repr(sep)}"
        )
        st.stop()

    with st.sidebar.expander("ℹ️ Diagnóstico del archivo", expanded=False):
        st.write(f"**Encoding:** `{encoding}`")
        st.write(f"**Separador:** `{repr(sep)}`")
        st.write(f"**Filas cargadas:** {len(df)}")
        st.write(f"**Sondas detectadas:** {len(sondas_activas)}")
        st.write(f"**Rango:** {df['Timestamp'].min()} → {df['Timestamp'].max()}")

    seleccionadas = st.multiselect(
        "Seleccionar Sondas", sondas_activas, default=sondas_activas
    )

    if seleccionadas:
        st.sidebar.header("Configuración del Gráfico")

        num_lecturas = st.sidebar.number_input(
            "Cantidad de lecturas a mostrar (0 = todas)",
            min_value=0, max_value=len(df), value=0,
        )

        min_t = df["Timestamp"].min().to_pydatetime()
        max_t = df["Timestamp"].max().to_pydatetime()

        rango = st.sidebar.slider(
            "Rango de tiempo",
            min_value=min_t, max_value=max_t, value=(min_t, max_t),
            format="DD/MM HH:mm:ss",
            step=pd.Timedelta(seconds=1).to_pytimedelta(),
        )

        df_f = df[(df["Timestamp"] >= rango[0]) & (df["Timestamp"] <= rango[1])]
        if num_lecturas > 0:
            df_f = df_f.tail(num_lecturas)

        if not df_f.empty:
            fig = go.Figure()
            for s in seleccionadas:
                df_p = df_f.dropna(subset=[s])
                if not df_p.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=df_p["Timestamp"],
                            y=df_p[s],
                            mode="lines+markers" if len(df_f) < 50 else "lines",
                            name=s,
                            connectgaps=True,
                        )
                    )

            fig.update_layout(
                title={"text": "GRÁFICO DE TEMPERATURA DEL EQUIPO",
                       "x": 0.5, "xanchor": "center"},
                template="plotly_white",
                xaxis_title="Tiempo",
                yaxis_title="Temperatura (°C)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Resumen Estadístico")
            bloque = df_f[seleccionadas]
            m1, m2, m3 = st.columns(3)
            m1.metric("MÁXIMO",  f"{bloque.max().max():.2f} °C")
            m2.metric("MÍNIMO",  f"{bloque.min().min():.2f} °C")
            m3.metric("PROMEDIO", f"{bloque.mean().mean():.2f} °C")
        else:
            st.warning("No hay datos para el rango seleccionado.")