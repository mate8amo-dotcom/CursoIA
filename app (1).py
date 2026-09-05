from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Explorador automático de datos",
    page_icon="📊",
    layout="wide",
)

FECHA_CLAVES = ("fecha", "date")


def normalizar_columnas(columnas: Iterable[object]) -> list[str]:
    """Limpia espacios en nombres y garantiza nombres únicos sin alterar los datos."""
    resultado: list[str] = []
    usados: dict[str, int] = {}
    for posicion, columna in enumerate(columnas, start=1):
        nombre = str(columna).strip() or f"columna_{posicion}"
        contador = usados.get(nombre, 0)
        usados[nombre] = contador + 1
        resultado.append(nombre if contador == 0 else f"{nombre}_{contador + 1}")
    return resultado


def convertir_fechas_por_nombre(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta convertir columnas candidatas a fecha; conserva la original si falla."""
    resultado = df.copy()
    for columna in resultado.columns:
        if any(clave in columna.lower() for clave in FECHA_CLAVES):
            original = resultado[columna]
            try:
                convertida = pd.to_datetime(original, errors="coerce")
                no_nulos = int(original.notna().sum())
                reconocidos = int(convertida.notna().sum())
                # Evita convertir columnas cuyo nombre coincide, pero cuyos valores no son fechas.
                if no_nulos == 0 or reconocidos / no_nulos >= 0.6:
                    resultado[columna] = convertida
            except (TypeError, ValueError, OverflowError):
                pass
    return resultado


@st.cache_data(show_spinner=False)
def leer_dataset(contenido: bytes, nombre_archivo: str) -> pd.DataFrame:
    """Lee CSV o Excel desde memoria y devuelve un DataFrame normalizado."""
    extension = nombre_archivo.lower().rsplit(".", 1)[-1]
    buffer = BytesIO(contenido)
    if extension == "csv":
        ultimo_error: Exception | None = None
        for codificacion in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding=codificacion, sep=None, engine="python")
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
                ultimo_error = error
        raise ValueError("No fue posible reconocer la codificación o el separador del CSV.") from ultimo_error
    if extension == "xlsx":
        return pd.read_excel(buffer, engine="openpyxl")
    if extension == "xls":
        return pd.read_excel(buffer, engine="xlrd")
    raise ValueError("Formato no admitido. Use CSV, XLSX o XLS.")


def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    resultado.columns = normalizar_columnas(resultado.columns)
    return convertir_fechas_por_nombre(resultado)


def tipo_analitico(serie: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(serie):
        return "Booleana"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "Fecha/hora"
    if pd.api.types.is_numeric_dtype(serie):
        return "Numérica"
    no_nulos = serie.dropna()
    unicos = int(no_nulos.nunique())
    if unicos == 0:
        return "Texto"
    proporcion = unicos / max(len(no_nulos), 1)
    return "Categórica" if unicos <= 50 or proporcion <= 0.2 else "Texto"


def clasificar_columnas(df: pd.DataFrame) -> dict[str, list[str]]:
    grupos = {"Numérica": [], "Categórica": [], "Texto": [], "Booleana": [], "Fecha/hora": []}
    for columna in df.columns:
        grupos[tipo_analitico(df[columna])].append(columna)
    return grupos


def resumen_tipos(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variable": df.columns,
            "Tipo Pandas": [str(df[c].dtype) for c in df.columns],
            "Tipo analítico": [tipo_analitico(df[c]) for c in df.columns],
            "Valores no nulos": [int(df[c].notna().sum()) for c in df.columns],
            "Valores únicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def tabla_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    cantidad = df.isna().sum()
    porcentaje = cantidad.div(len(df)).mul(100) if len(df) else cantidad.astype(float)
    return (
        pd.DataFrame({"Variable": df.columns, "Valores faltantes": cantidad.values,
                      "Porcentaje faltante": porcentaje.values})
        .sort_values(["Valores faltantes", "Variable"], ascending=[False, True])
        .reset_index(drop=True)
    )


def csv_descargable(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def estadisticas_numericas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    if not columnas:
        raise ValueError("El conjunto filtrado no contiene variables numéricas.")
    return df[columnas].describe().rename(
        index={"count": "Conteo", "mean": "Media", "std": "Desviación estándar",
               "min": "Mínimo", "25%": "Primer cuartil", "50%": "Mediana",
               "75%": "Tercer cuartil", "max": "Máximo"}
    )


def estadisticas_categoricas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    if not columnas:
        raise ValueError("El conjunto filtrado no contiene variables categóricas, de texto o booleanas.")
    filas = []
    for columna in columnas:
        serie = df[columna].dropna()
        frecuencias = serie.value_counts(dropna=True)
        filas.append({
            "Variable": columna,
            "Conteo": int(serie.count()),
            "Valores únicos": int(serie.nunique()),
            "Categoría más frecuente": frecuencias.index[0] if not frecuencias.empty else np.nan,
            "Frecuencia dominante": int(frecuencias.iloc[0]) if not frecuencias.empty else 0,
        })
    return pd.DataFrame(filas)


def detectar_atipicos(df: pd.DataFrame, columnas: list[str], factor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    hallazgos: list[pd.DataFrame] = []
    resumen: list[dict[str, object]] = []
    for columna in columnas:
        serie = df[columna]
        validos = serie.dropna()
        if validos.empty:
            inferior = superior = np.nan
            mascara = pd.Series(False, index=df.index)
        else:
            q1, q3 = validos.quantile([0.25, 0.75])
            iqr = q3 - q1
            inferior = q1 - factor * iqr
            superior = q3 + factor * iqr
            mascara = serie.lt(inferior) | serie.gt(superior)
        detecciones = int(mascara.sum())
        resumen.append({"Variable": columna, "Cantidad de atípicos": detecciones})
        if detecciones:
            detalle = df.loc[mascara].copy()
            detalle.insert(0, "Fila original", df.index[mascara])
            detalle.insert(1, "Variable atípica", columna)
            detalle.insert(2, "Valor atípico", serie.loc[mascara].values)
            detalle.insert(3, "Límite inferior", inferior)
            detalle.insert(4, "Límite superior", superior)
            hallazgos.append(detalle)
    detalle_total = pd.concat(hallazgos, ignore_index=True) if hallazgos else pd.DataFrame(
        columns=["Fila original", "Variable atípica", "Valor atípico", "Límite inferior", "Límite superior"] + list(df.columns)
    )
    return detalle_total, pd.DataFrame(resumen)


def aplicar_filtros(df: pd.DataFrame, grupos: dict[str, list[str]]) -> pd.DataFrame:
    resultado = df.copy()
    with st.sidebar.expander("Filtros interactivos", expanded=True):
        st.caption("Los valores faltantes se conservan en los filtros de fecha y numéricos.")

        for columna in grupos["Fecha/hora"]:
            serie = resultado[columna]
            validos = serie.dropna()
            if validos.empty:
                st.caption(f"{columna}: sin fechas válidas para filtrar.")
                continue
            minimo, maximo = validos.min().date(), validos.max().date()
            rango = st.date_input(
                f"Rango de {columna}", value=(minimo, maximo), min_value=minimo,
                max_value=maximo, key=f"fecha_{columna}"
            )
            if isinstance(rango, (tuple, list)) and len(rango) == 2:
                inicial, final = rango
                fechas = resultado[columna].dt.date
                resultado = resultado[resultado[columna].isna() | fechas.between(inicial, final)]

        candidatas_cat = grupos["Categórica"] + grupos["Booleana"]
        filtros_cat = st.multiselect("Variables categóricas para filtrar", candidatas_cat)
        for columna in filtros_cat:
            opciones = df[columna].dropna().unique().tolist()
            seleccion = st.multiselect(f"Categorías de {columna}", opciones, default=opciones,
                                       key=f"cat_{columna}")
            if seleccion:
                resultado = resultado[resultado[columna].isin(seleccion)]
            else:
                resultado = resultado.iloc[0:0]

        filtros_num = st.multiselect("Variables numéricas para filtrar", grupos["Numérica"])
        for columna in filtros_num:
            validos = df[columna].dropna()
            if validos.empty:
                st.caption(f"{columna}: sin valores numéricos para filtrar.")
                continue
            minimo, maximo = float(validos.min()), float(validos.max())
            if minimo == maximo:
                st.caption(f"{columna}: valor constante ({minimo:g}).")
                continue
            paso = max((maximo - minimo) / 100, np.finfo(float).eps)
            inferior, superior = st.slider(
                f"Rango de {columna}", min_value=minimo, max_value=maximo,
                value=(minimo, maximo), step=paso, key=f"num_{columna}"
            )
            resultado = resultado[resultado[columna].isna() | resultado[columna].between(inferior, superior)]
    return resultado


st.title("📊 Explorador automático de datos")
st.write("Cargue un archivo y obtenga un análisis exploratorio interactivo, sin depender de un conjunto de datos predeterminado.")

with st.sidebar:
    st.header("Carga de datos")
    archivo = st.file_uploader("Seleccione un archivo", type=["csv", "xlsx", "xls"])

if archivo is None:
    st.info("Cargue un archivo desde la barra lateral para iniciar el análisis.")
    col1, col2, col3 = st.columns(3)
    col1.markdown("### 1. Cargar\nFormatos permitidos: **CSV, XLSX y XLS**.")
    col2.markdown("### 2. Explorar\nRevise calidad, estadísticas, distribuciones, correlaciones y atípicos.")
    col3.markdown("### 3. Descargar\nExporte los datos filtrados y las detecciones de atípicos.")
    st.markdown("""
    **Análisis disponibles:** indicadores generales, tipos de variables, duplicados, valores faltantes,
    estadísticas descriptivas, distribuciones, correlaciones, detección IQR, filtros y tabla ordenable.
    """)
    st.warning("Los datos se procesan durante la sesión. Evite cargar información personal, confidencial o sensible.")
    st.stop()

try:
    bruto = leer_dataset(archivo.getvalue(), archivo.name)
    if bruto.empty or bruto.shape[1] == 0:
        st.warning("El archivo está vacío o no contiene una tabla utilizable.")
        st.stop()
    df_original = preparar_dataset(bruto)
except Exception as error:
    st.error(f"No fue posible procesar el archivo. Verifique su formato y contenido. Detalle: {error}")
    st.stop()

st.sidebar.success(f"Archivo cargado: {archivo.name}")
grupos_originales = clasificar_columnas(df_original)
df = aplicar_filtros(df_original, grupos_originales)
st.sidebar.metric("Registros después de filtros", len(df))

if df.empty:
    st.warning("Los filtros no producen registros. Ajuste los filtros de la barra lateral para continuar.")
    st.stop()

grupos = clasificar_columnas(df)
duplicados = int(df.duplicated().sum())
faltantes = int(df.isna().sum().sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Filas", f"{df.shape[0]:,}")
m2.metric("Columnas", f"{df.shape[1]:,}")
m3.metric("Duplicados completos", f"{duplicados:,}")
m4.metric("Celdas faltantes", f"{faltantes:,}")
st.caption(f"Archivo: **{archivo.name}** · Dimensiones actuales: **{df.shape[0]} filas × {df.shape[1]} columnas**")

pestanas = st.tabs([
    "Resumen y tipos", "Calidad de datos", "Estadísticas", "Distribuciones",
    "Correlaciones", "Valores atípicos", "Tabla ordenable"
])

with pestanas[0]:
    st.subheader("Tipos de variables")
    st.dataframe(resumen_tipos(df), use_container_width=True, hide_index=True)

with pestanas[1]:
    st.subheader("Registros duplicados")
    involucrados = df[df.duplicated(keep=False)]
    if involucrados.empty:
        st.success("No se encontraron registros completamente duplicados.")
    else:
        st.warning(f"Se encontraron {duplicados} duplicados adicionales y {len(involucrados)} registros involucrados.")
        st.dataframe(involucrados, use_container_width=True, hide_index=True)

    st.subheader("Valores faltantes")
    faltantes_df = tabla_faltantes(df)
    st.dataframe(
        faltantes_df.style.format({"Porcentaje faltante": "{:.2f}%"}),
        use_container_width=True, hide_index=True
    )
    grafico_faltantes = px.bar(
        faltantes_df, x="Variable", y="Porcentaje faltante",
        title="Porcentaje de valores faltantes por variable",
        labels={"Porcentaje faltante": "Porcentaje (%)"}
    )
    st.plotly_chart(grafico_faltantes, use_container_width=True)

with pestanas[2]:
    st.subheader("Estadísticas descriptivas")
    opcion = st.radio(
        "Variables que desea resumir",
        ["Todas las variables", "Solo variables numéricas", "Solo variables categóricas"],
        horizontal=True
    )
    columnas_cat = grupos["Categórica"] + grupos["Texto"] + grupos["Booleana"]
    try:
        if opcion in ("Todas las variables", "Solo variables numéricas"):
            st.markdown("#### Variables numéricas")
            st.dataframe(estadisticas_numericas(df, grupos["Numérica"]), use_container_width=True)
        if opcion in ("Todas las variables", "Solo variables categóricas"):
            st.markdown("#### Variables categóricas, de texto y booleanas")
            st.dataframe(estadisticas_categoricas(df, columnas_cat), use_container_width=True, hide_index=True)
    except ValueError as error:
        st.info(str(error))
    except Exception as error:
        st.error(f"No fue posible calcular las estadísticas: {error}")

with pestanas[3]:
    st.subheader("Distribuciones")
    candidatas = grupos["Numérica"] + grupos["Categórica"] + grupos["Texto"] + grupos["Booleana"]
    if not candidatas:
        st.info("No hay variables compatibles para mostrar distribuciones.")
    else:
        variable = st.selectbox("Seleccione una variable", candidatas)
        if variable in grupos["Numérica"]:
            intervalos = st.slider("Número de intervalos", 5, 100, 30)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(df, x=variable, nbins=intervalos, title=f"Histograma de {variable}"),
                            use_container_width=True)
            agrupadoras = ["Sin agrupación"] + grupos["Categórica"] + grupos["Booleana"]
            agrupadora = st.selectbox("Agrupar diagrama de caja por", agrupadoras)
            color = None if agrupadora == "Sin agrupación" else agrupadora
            fig_caja = px.box(df, x=color, y=variable, points="outliers", title=f"Diagrama de caja de {variable}")
            c2.plotly_chart(fig_caja, use_container_width=True)
        else:
            etiquetas = df[variable].astype("object").where(df[variable].notna(), "(Faltante)").astype(str)
            frecuencias = etiquetas.value_counts().head(30).rename_axis("Categoría").reset_index(name="Frecuencia")
            if etiquetas.nunique() > 30:
                st.info("Se muestran las 30 categorías más frecuentes.")
            st.plotly_chart(px.bar(frecuencias, x="Categoría", y="Frecuencia",
                                   title=f"Frecuencia de {variable}"), use_container_width=True)

with pestanas[4]:
    st.subheader("Correlaciones")
    numericas = grupos["Numérica"]
    seleccion = st.multiselect("Variables incluidas", numericas, default=numericas)
    metodo_visible = st.selectbox("Método", ["Pearson", "Spearman", "Kendall"])
    if len(seleccion) < 2:
        st.info("Seleccione al menos dos variables numéricas.")
    else:
        matriz = df[seleccion].corr(method=metodo_visible.lower())
        fig = go.Figure(data=go.Heatmap(
            z=matriz.values, x=matriz.columns, y=matriz.index,
            zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            text=np.round(matriz.values, 2), texttemplate="%{text:.2f}",
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"
        ))
        fig.update_layout(title=f"Correlación de {metodo_visible}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(matriz.style.format("{:.3f}"), use_container_width=True)
    st.caption("Una correlación no implica causalidad.")

with pestanas[5]:
    st.subheader("Valores atípicos por rango intercuartílico")
    seleccion_atipicos = st.multiselect("Variables numéricas", grupos["Numérica"], default=grupos["Numérica"])
    factor = st.slider("Factor IQR", 1.0, 3.0, 1.5, 0.1)
    detalle_atipicos, resumen_atipicos = detectar_atipicos(df, seleccion_atipicos, factor)
    st.metric("Detecciones", len(detalle_atipicos))
    if seleccion_atipicos:
        st.plotly_chart(px.bar(resumen_atipicos, x="Variable", y="Cantidad de atípicos",
                               title="Cantidad de atípicos por variable"), use_container_width=True)
    if detalle_atipicos.empty:
        st.success("No se detectaron valores atípicos con la selección y el factor actuales.")
    else:
        st.dataframe(detalle_atipicos, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar valores atípicos", data=csv_descargable(detalle_atipicos),
        file_name="valores_atipicos.csv", mime="text/csv", disabled=not seleccion_atipicos
    )
    st.caption("Un valor atípico no necesariamente representa un error.")

with pestanas[6]:
    st.subheader("Tabla interactiva y ordenable")
    visibles = st.multiselect("Columnas visibles", list(df.columns), default=list(df.columns))
    if not visibles:
        st.info("Seleccione al menos una columna para mostrar la tabla.")
    else:
        st.dataframe(df[visibles], use_container_width=True, hide_index=True, height=520)
    st.download_button(
        "Descargar datos filtrados", data=csv_descargable(df),
        file_name="datos_filtrados.csv", mime="text/csv"
    )

st.divider()
st.info(
    "🔒 Los datos se procesan durante la sesión de la aplicación. Evite cargar información personal, "
    "confidencial o sensible. Este análisis exploratorio no reemplaza la interpretación experta. "
    "Una correlación no implica causalidad y un valor atípico no necesariamente representa un error."
)
