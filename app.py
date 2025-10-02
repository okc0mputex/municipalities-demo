import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# Absolute default path to the provided Excel file
DEFAULT_XLSX_PATH = "/Users/vasilis/1/Στοιχεία Συμβάσεων_Demo Έκδοση.xlsx"
DEFAULT_CSV_DIR = "/Users/vasilis/1/clean_csv"


def _normalize_column_name(column_name: str) -> str:
    """Return a casefolded version of the column name for robust matching (handles Greek/Latin)."""
    return str(column_name).strip().casefold()


def _find_first(items: List[str], predicate) -> Optional[str]:
    for item in items:
        if predicate(item):
            return item
    return None


def infer_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Heuristically infer key semantic columns.

    Returns a mapping with keys: "date", "amount", "customer", "status", "contractor", "payment_date", "payment_amount", "category".
    Each value is the column name from df.columns or None if not inferred.
    """
    normalized_by_original = {col: _normalize_column_name(col) for col in df.columns}
    normalized_to_original = {v: k for k, v in normalized_by_original.items()}

    # Keyword candidates (English + Greek stems)
    date_keywords = [
        "date", "dt", "ημερο", "ημερ", "ημ/νια", "ημ/νία", "ημερομην", "date_signed", "startdate", "enddate",
    ]
    amount_keywords = [
        "amount", "value", "total", "price", "cost", "ποσ", "συνολ", "κοστος", "αξια", "αξία",
    ]
    customer_keywords = [
        "customer", "client", "account", "buyer", "company", "vendor", "πελατ", "πελάτ", "εταιρ", "εταιρεία",
    ]
    contractor_keywords = [
        "contractor", "supplier", "vendor", "ανάδοχ", "προμηθευ", "εργολ", "αναδοχ",
    ]
    category_keywords = [
        "category", "κατηγορ", "είδος", "τύπος",
    ]
    status_keywords = [
        "status", "state", "stage", "phase", "καταστα", "στάδ", "σταδ",
    ]
    payment_date_keywords = [
        "payment date", "paid date", "ημερ πληρω", "ημερομηνία πληρω", "πληρωμ", "καταβολ", "εξόφλ"
    ]
    payment_amount_keywords = [
        "payment amount", "amount paid", "paid", "πληρωμ ποσ", "ποσό πληρω", "εξοφληθ", "καταβληθ"
    ]

    # Type-based hints
    datetime_columns = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    numeric_columns = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    def pick_by_keywords(keywords: List[str]) -> Optional[str]:
        for key in keywords:
            match = _find_first(list(normalized_to_original.keys()), lambda n: key in n)
            if match is not None:
                return normalized_to_original[match]
        return None

    inferred_date = None
    # Prefer actual datetime dtype, else keyword
    if datetime_columns:
        inferred_date = datetime_columns[0]
    else:
        inferred_date = pick_by_keywords(date_keywords)

    inferred_amount = None
    if numeric_columns:
        # Try numeric column that matches amount-like keywords first
        amt_by_kw = pick_by_keywords(amount_keywords)
        if amt_by_kw in numeric_columns:
            inferred_amount = amt_by_kw
        else:
            # fallback to the first numeric column
            inferred_amount = numeric_columns[0]
    else:
        amt_by_kw = pick_by_keywords(amount_keywords)
        inferred_amount = amt_by_kw

    inferred_customer = pick_by_keywords(customer_keywords)
    inferred_contractor = pick_by_keywords(contractor_keywords) or inferred_customer
    inferred_status = pick_by_keywords(status_keywords)
    inferred_category = pick_by_keywords(category_keywords)
    inferred_payment_date = pick_by_keywords(payment_date_keywords)

    inferred_payment_amount = None
    if numeric_columns:
        pay_by_kw = pick_by_keywords(payment_amount_keywords)
        if pay_by_kw in numeric_columns:
            inferred_payment_amount = pay_by_kw
        else:
            inferred_payment_amount = None
    else:
        inferred_payment_amount = pick_by_keywords(payment_amount_keywords)

    return {
        "date": inferred_date,
        "amount": inferred_amount,
        "customer": inferred_customer,
        "contractor": inferred_contractor,
        "status": inferred_status,
        "payment_date": inferred_payment_date,
        "payment_amount": inferred_payment_amount,
        "category": inferred_category,
    }


@st.cache_data(show_spinner=False)
def list_sheet_names(source: Union[io.BytesIO, str, Path]) -> List[str]:
    xls = pd.ExcelFile(source)
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def load_excel_preview(source: Union[io.BytesIO, str, Path], sheet_name: Optional[str] = None, nrows: int = 20) -> pd.DataFrame:
    """Load a small preview without headers to help the user pick the header row."""
    df = pd.read_excel(source, sheet_name=sheet_name, header=None, nrows=nrows)
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]
    return df


@st.cache_data(show_spinner=False)
def load_excel_dataframe(
    source: Union[io.BytesIO, str, Path],
    sheet_name: Optional[str] = None,
    header_start_row_index_1based: Optional[int] = None,
    header_depth: int = 1,
) -> pd.DataFrame:
    """Load Excel with configurable header start row (1-based) and header depth (rows to combine)."""
    header_param: Union[int, List[int], None]
    if header_start_row_index_1based is None:
        header_param = 0
    else:
        start0 = max(0, int(header_start_row_index_1based) - 1)
        if header_depth and header_depth > 1:
            header_param = list(range(start0, start0 + header_depth))
        else:
            header_param = start0

    df = pd.read_excel(source, sheet_name=sheet_name, header=header_param)
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]
    return df


@st.cache_data(show_spinner=False)
def list_csv_files(directory: Union[str, Path]) -> List[str]:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []
    return [str(p) for p in sorted(directory.glob("*.csv"))]


@st.cache_data(show_spinner=False)
def load_csv_dataframe(source: Union[io.BytesIO, str, Path]) -> pd.DataFrame:
    return pd.read_csv(source, encoding="utf-8-sig")


def _flatten_and_clean_columns(columns: pd.Index) -> List[str]:
    """Flatten MultiIndex columns and clean placeholders like 'Unnamed: x' and whitespace.

    Ensures resulting names are unique by appending numeric suffixes when needed.
    """
    def as_str(value: object) -> str:
        if value is None:
            return ""
        s = str(value)
        return s

    flattened: List[str] = []
    for col in columns:
        parts: List[str]
        if isinstance(col, tuple):
            parts = [p for p in (as_str(x).strip() for x in col) if p and not p.startswith("Unnamed:")]
        else:
            s = as_str(col).strip()
            parts = [] if (not s or s.startswith("Unnamed:")) else [s]
        name = " - ".join(parts).strip()
        if not name:
            name = "column"
        flattened.append(name)

    # Ensure uniqueness
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for name in flattened:
        if name not in seen:
            seen[name] = 1
            unique.append(name)
        else:
            seen[name] += 1
            unique.append(f"{name} ({seen[name]})")
    return unique


def _suggest_header_row_index_1based(preview_df: pd.DataFrame) -> int:
    """Heuristic: pick the row (1..N) with the most non-null string-like cells among first N preview rows."""
    best_row = 1
    best_score = -1
    rows_to_check = min(len(preview_df), 20)
    for i in range(rows_to_check):
        row = preview_df.iloc[i]
        score = int(row.notna().sum())
        if score > best_score:
            best_score = score
            best_row = i + 1  # 1-based
    return best_row


def _smart_parse_datetime(series: pd.Series) -> pd.Series:
    """Parse dates from mixed formats, including:
    - ISO/date strings (dayfirst tolerant)
    - Excel serial day numbers (origin 1899-12-30)
    - Unix epoch seconds/milliseconds
    Filters out unrealistic years (<1990 or >2100) as NaT.
    """
    s = series.copy()
    # First pass: try direct parse from strings
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # Handle numeric-like separately
    num = pd.to_numeric(s, errors="coerce")
    # Excel serial days heuristic
    as_excel = pd.to_datetime(num, errors="coerce", unit="D", origin="1899-12-30")
    # Epoch seconds/milliseconds heuristics
    as_epoch_sec = pd.to_datetime(num, errors="coerce", unit="s")
    as_epoch_ms = pd.to_datetime(num, errors="coerce", unit="ms")

    # Choose best candidate per row: prefer existing parsed, else excel if not NaT, else epoch sec, else epoch ms
    result = parsed.where(parsed.notna(), as_excel)
    result = result.where(result.notna(), as_epoch_sec)
    result = result.where(result.notna(), as_epoch_ms)

    # Drop unrealistic years
    valid_mask = result.notna()
    with pd.option_context('mode.use_inf_as_na', True):
        years = result.dt.year.where(valid_mask, other=np.nan)
    valid_years = years.between(1990, 2100, inclusive="both")
    result = result.where(valid_years, other=pd.NaT)
    return result


def coerce_datetime(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col and date_col in df.columns:
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df = df.copy()
            df[date_col] = _smart_parse_datetime(df[date_col])
    return df


def apply_global_filters(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df

    date_col = mapping.get("date")
    amount_col = mapping.get("amount")
    customer_col = mapping.get("customer")
    status_col = mapping.get("status")

    # Date range filter
    if date_col and date_col in filtered.columns:
        filtered = coerce_datetime(filtered, date_col)
        valid_dates = filtered[date_col].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            start_date, end_date = st.sidebar.date_input(
                "Ημερομηνία (εύρος)", value=(min_date, max_date), min_value=min_date, max_value=max_date
            )
            if isinstance(start_date, tuple):  # streamlit may return tuple if not properly set
                start_date, end_date = start_date
            mask = (filtered[date_col] >= pd.to_datetime(start_date)) & (filtered[date_col] <= pd.to_datetime(end_date))
            filtered = filtered.loc[mask]

    # Customer filter
    if customer_col and customer_col in filtered.columns:
        customers = sorted([c for c in filtered[customer_col].dropna().astype(str).unique()])
        selected_customers = st.sidebar.multiselect("Πελάτης", customers, default=customers)
        if selected_customers:
            filtered = filtered[filtered[customer_col].astype(str).isin(selected_customers)]

    # Status filter
    if status_col and status_col in filtered.columns:
        statuses = sorted([s for s in filtered[status_col].dropna().astype(str).unique()])
        selected_statuses = st.sidebar.multiselect("Κατάσταση", statuses, default=statuses)
        if selected_statuses:
            filtered = filtered[filtered[status_col].astype(str).isin(selected_statuses)]

    # Amount filter
    if amount_col and amount_col in filtered.columns and np.issubdtype(filtered[amount_col].dtype, np.number):
        min_amt = float(np.nanmin(filtered[amount_col].values)) if not filtered.empty else 0.0
        max_amt = float(np.nanmax(filtered[amount_col].values)) if not filtered.empty else 0.0
        min_sel, max_sel = st.sidebar.slider("Ποσό", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt))
        filtered = filtered[(filtered[amount_col] >= min_sel) & (filtered[amount_col] <= max_sel)]

    return filtered


def kpi_box(label: str, value: Union[float, int, str], help_text: Optional[str] = None):
    st.metric(label=label, value=value, help=help_text)


def render_overall_tab(df: pd.DataFrame, mapping: Dict[str, Optional[str]]):
    date_col = mapping.get("date")
    amount_col = mapping.get("amount")
    contractor_col = mapping.get("contractor")
    status_col = mapping.get("status")
    category_col = mapping.get("category")

    cols = st.columns(3)
    total_rows = int(len(df))
    with cols[0]:
        kpi_box("Σύνολο εγγραφών", total_rows)

    if amount_col and amount_col in df.columns and np.issubdtype(df[amount_col].dtype, np.number):
        total_amount = float(df[amount_col].sum())
        avg_amount = float(df[amount_col].mean()) if total_rows else 0.0
        with cols[1]:
            kpi_box("Συνολικό ποσό", f"{total_amount:,.2f}")
        with cols[2]:
            kpi_box("Μέσο ποσό", f"{avg_amount:,.2f}")

    st.divider()

    if date_col and date_col in df.columns:
        df_dt = coerce_datetime(df, date_col).dropna(subset=[date_col])
        if not df_dt.empty:
            # Group by month
            df_dt = df_dt.copy()
            df_dt["__month"] = df_dt[date_col].dt.to_period("M").dt.to_timestamp()
            if amount_col and amount_col in df_dt.columns and np.issubdtype(df_dt[amount_col].dtype, np.number):
                series = df_dt.groupby("__month")[amount_col].sum().reset_index()
                fig = px.line(series, x="__month", y=amount_col, title="Ποσό ανά μήνα")
            else:
                series = df_dt.groupby("__month").size().reset_index(name="count")
                fig = px.line(series, x="__month", y="count", title="Εγγραφές ανά μήνα")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Status breakdown
    if status_col and status_col in df.columns:
        st.subheader("Κατανομή Κατάστασης")
        if amount_col and amount_col in df.columns and np.issubdtype(df[amount_col].dtype, np.number):
            grouped = df.groupby(status_col)[amount_col].sum().reset_index()
            fig = px.bar(grouped, x=status_col, y=amount_col, title="Ποσό ανά κατάσταση")
        else:
            grouped = df.groupby(status_col).size().reset_index(name="count")
            fig = px.pie(grouped, names=status_col, values="count", title="Πλήθος εγγραφών ανά κατάσταση")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Top contractors by contract amount
    if contractor_col and contractor_col in df.columns:
        st.subheader("Top Ανάδοχοι (Συμβάσεις)")
        top_n = st.slider("Top N", 5, 50, 10, key="overall_topn")
        if amount_col and amount_col in df.columns and np.issubdtype(df[amount_col].dtype, np.number):
            grouped = (
                df.groupby(contractor_col)[amount_col]
                .sum()
                .reset_index()
                .sort_values(amount_col, ascending=False)
                .head(top_n)
            )
            fig = px.bar(grouped, x=contractor_col, y=amount_col, title="Ποσό Συμβάσεων ανά Ανάδοχο")
        else:
            grouped = df.groupby(contractor_col).size().reset_index(name="count").sort_values("count", ascending=False).head(top_n)
            fig = px.bar(grouped, x=contractor_col, y="count", title="Πλήθος Συμβάσεων ανά Ανάδοχο")
        fig.update_layout(xaxis_title=None, margin=dict(l=0, r=0, t=40, b=80))
        st.plotly_chart(fig, use_container_width=True)

    # Category breakdown and top vendors by category
    if category_col and category_col in df.columns:
        st.subheader("Κατανομή Κατηγορίας Συμβάσεων")
        if amount_col and amount_col in df.columns and np.issubdtype(df[amount_col].dtype, np.number):
            grouped = df.groupby(category_col)[amount_col].sum().reset_index()
            fig = px.bar(grouped, x=category_col, y=amount_col, title="Ποσό ανά Κατηγορία")
        else:
            grouped = df.groupby(category_col).size().reset_index(name="count")
            fig = px.bar(grouped, x=category_col, y="count", title="Πλήθος ανά Κατηγορία")
        fig.update_layout(xaxis_title=None, margin=dict(l=0, r=0, t=40, b=80))
        st.plotly_chart(fig, use_container_width=True)

        if contractor_col and contractor_col in df.columns and amount_col and amount_col in df.columns and np.issubdtype(df[amount_col].dtype, np.number):
            st.subheader("Top Ανάδοχοι ανά Κατηγορία")
            chosen_cat = st.selectbox("Κατηγορία", sorted([c for c in df[category_col].dropna().astype(str).unique()]))
            top_n2 = st.slider("Top N (ανά κατηγορία)", 3, 20, 5, key="overall_topn_cat")
            sub = df[df[category_col].astype(str) == chosen_cat]
            by_vendor = (
                sub.groupby(contractor_col)[amount_col]
                .sum()
                .reset_index()
                .sort_values(amount_col, ascending=False)
                .head(top_n2)
            )
            fig = px.bar(by_vendor, x=contractor_col, y=amount_col, title=f"{chosen_cat}: Ποσό συμβάσεων ανά ανάδοχο")
            fig.update_layout(xaxis_title=None, margin=dict(l=0, r=0, t=40, b=80))
            st.plotly_chart(fig, use_container_width=True)


def render_payments_tab(df: pd.DataFrame, mapping: Dict[str, Optional[str]]):
    payment_amount_col = mapping.get("payment_amount")
    payment_date_col = mapping.get("payment_date")
    contractor_col = mapping.get("contractor")
    status_col = mapping.get("status")
    amount_col = mapping.get("amount")

    # Fallback selectors if missing
    if not payment_amount_col or payment_amount_col not in df.columns:
        st.info("Δεν βρέθηκε στήλη ποσού πληρωμής.")
        candidate_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        payment_amount_col = st.selectbox("Επιλέξτε στήλη ποσού πληρωμής", options=candidate_numeric if candidate_numeric else list(df.columns))
    if not payment_date_col or payment_date_col not in df.columns:
        candidates = list(df.columns)
        payment_date_col = st.selectbox("Επιλέξτε στήλη ημερομηνίας πληρωμής (προαιρετικό)", options=[None] + candidates, index=0)

    payments_df = df.copy()
    # Ensure numeric
    payments_df[payment_amount_col] = pd.to_numeric(payments_df[payment_amount_col], errors="coerce")
    # Coerce payment date if available
    if payment_date_col and payment_date_col in payments_df.columns:
        payments_df = coerce_datetime(payments_df, payment_date_col)

    # KPIs
    cols = st.columns(4)
    valid_payment_mask = payments_df[payment_amount_col].notna()
    total_payments = int(valid_payment_mask.sum())
    total_paid = float(payments_df.loc[valid_payment_mask, payment_amount_col].sum())
    avg_payment = float(payments_df.loc[valid_payment_mask, payment_amount_col].mean()) if total_payments else 0.0
    with cols[0]:
        kpi_box("Πλήθος πληρωμών", total_payments)
    with cols[1]:
        kpi_box("Συνολικό καταβεβλημένο", f"{total_paid:,.2f}")
    with cols[2]:
        kpi_box("Μέση πληρωμή", f"{avg_payment:,.2f}")
    if amount_col and amount_col in payments_df.columns and pd.api.types.is_numeric_dtype(payments_df[amount_col]):
        coverage = float(100.0 * total_paid / payments_df[amount_col].sum()) if payments_df[amount_col].sum() else 0.0
        with cols[3]:
            kpi_box("Κάλυψη πληρωμών %", f"{coverage:,.1f}%")

    st.divider()

    # Monthly trend
    if payment_date_col and payment_date_col in payments_df.columns and payments_df[payment_date_col].notna().any():
        payments_df = payments_df.dropna(subset=[payment_date_col])
        payments_df["__month"] = payments_df[payment_date_col].dt.to_period("M").dt.to_timestamp()
        monthly = payments_df.groupby("__month")[payment_amount_col].sum().reset_index()
        fig = px.area(monthly, x="__month", y=payment_amount_col, title="Πληρωμές ανά μήνα")
        fig.update_traces(mode="lines")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Top contractors by paid amount
    if contractor_col and contractor_col in payments_df.columns:
        st.subheader("Top Ανάδοχοι κατά Πληρωμές")
        top_n = st.slider("Top N", 5, 50, 10, key="pay_topn")
        by_vendor = (
            payments_df.groupby(contractor_col)[payment_amount_col]
            .sum()
            .reset_index()
            .sort_values(payment_amount_col, ascending=False)
            .head(top_n)
        )
        fig = px.bar(by_vendor, x=contractor_col, y=payment_amount_col, title="Πληρωμές ανά Ανάδοχο")
        fig.update_layout(xaxis_title=None, margin=dict(l=0, r=0, t=40, b=80))
        st.plotly_chart(fig, use_container_width=True)

    # Payments by status
    if status_col and status_col in payments_df.columns:
        st.subheader("Πληρωμές ανά Κατάσταση")
        by_status = payments_df.groupby(status_col)[payment_amount_col].sum().reset_index()
        fig = px.bar(by_status, x=status_col, y=payment_amount_col, title="Ποσά ανά Κατάσταση")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Recent payments table
    st.subheader("Πρόσφατες Πληρωμές")
    recent = payments_df[[c for c in [payment_date_col, contractor_col, payment_amount_col, status_col] if c and c in payments_df.columns]].copy()
    if payment_date_col and payment_date_col in recent.columns:
        recent = recent.sort_values(payment_date_col, ascending=False)
    st.dataframe(recent.head(50), use_container_width=True)


def render_contractor_tab(df: pd.DataFrame, mapping: Dict[str, Optional[str]]):
    contractor_col = mapping.get("contractor")
    amount_col = mapping.get("amount")
    payment_amount_col = mapping.get("payment_amount")
    payment_date_col = mapping.get("payment_date")
    status_col = mapping.get("status")
    if not contractor_col or contractor_col not in df.columns:
        st.info("Δεν βρέθηκε στήλη αναδόχου/προμηθευτή.")
        return

    vendors = sorted([v for v in df[contractor_col].dropna().astype(str).unique()])
    selected_vendor = st.selectbox("Ανάδοχος/Προμηθευτής", vendors, index=0 if vendors else None, help="Επιλέξτε ανάδοχο για σύνοψη και λεπτομέρειες")
    if not selected_vendor:
        return
    sub = df[df[contractor_col].astype(str) == selected_vendor]

    # KPIs
    cols = st.columns(4)
    num_contracts = int(len(sub))
    total_amount = float(sub[amount_col].sum()) if amount_col and amount_col in sub.columns and np.issubdtype(sub[amount_col].dtype, np.number) else 0.0
    total_paid = float(sub[payment_amount_col].sum()) if payment_amount_col and payment_amount_col in sub.columns and np.issubdtype(sub[payment_amount_col].dtype, np.number) else 0.0
    outstanding = max(0.0, total_amount - total_paid) if total_amount else 0.0
    pct_paid = (total_paid / total_amount * 100.0) if total_amount else 0.0
    with cols[0]:
        kpi_box("Συμβάσεις (πλήθος)", num_contracts)
    with cols[1]:
        kpi_box("Συνολικό ποσό σύμβασης", f"{total_amount:,.2f}")
    with cols[2]:
        kpi_box("Σύνολο πληρωμών", f"{total_paid:,.2f}")
    with cols[3]:
        kpi_box("Υπόλοιπο", f"{outstanding:,.2f}")

    st.progress(min(1.0, pct_paid / 100.0), text=f"Ποσοστό εξόφλησης: {pct_paid:,.1f}%")

    # Payments timeline for this contractor (cumulative vs total contract amount)
    if payment_date_col and payment_date_col in sub.columns and payment_amount_col and payment_amount_col in sub.columns:
        sub = coerce_datetime(sub, payment_date_col)
        sub_pay = sub.dropna(subset=[payment_date_col])
        if not sub_pay.empty:
            sub_pay = sub_pay.copy()
            sub_pay["__month"] = sub_pay[payment_date_col].dt.to_period("M").dt.to_timestamp()
            monthly = sub_pay.groupby("__month")[payment_amount_col].sum().reset_index()
            monthly = monthly.sort_values("__month")
            monthly["cumulative_paid"] = monthly[payment_amount_col].cumsum()
            # Total contract amount as reference line
            total_line = total_amount if total_amount else monthly["cumulative_paid"].max()
            fig = px.line(monthly, x="__month", y="cumulative_paid", title="Συσσωρευτικές πληρωμές (ανάδοχος)")
            fig.add_hline(y=total_line, line_dash="dash", annotation_text="Συνολικό ποσό σύμβασης", annotation_position="bottom right")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Status breakdown for this contractor
    if status_col and status_col in sub.columns:
        st.subheader("Κατανομή Κατάστασης (ανάδοχος)")
        if amount_col and amount_col in sub.columns and np.issubdtype(sub[amount_col].dtype, np.number):
            grouped = sub.groupby(status_col)[amount_col].sum().reset_index()
            fig = px.bar(grouped, x=status_col, y=amount_col, title="Ποσό ανά κατάσταση")
        else:
            grouped = sub.groupby(status_col).size().reset_index(name="count")
            fig = px.pie(grouped, names=status_col, values="count", title="Πλήθος εγγραφών ανά κατάσταση")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Details table with common fields
    st.subheader("Λεπτομέρειες")
    cols_show = [c for c in [mapping.get("date"), amount_col, payment_date_col, payment_amount_col, status_col, contractor_col] if c and c in sub.columns]
    details = sub[cols_show] if cols_show else sub
    st.dataframe(details, use_container_width=True)
    st.download_button("Λήψη CSV (ανάδοχος)", data=details.to_csv(index=False).encode("utf-8-sig"), file_name=f"contractor_{selected_vendor}.csv", mime="text/csv")


def render_timeline_tab(df: pd.DataFrame, mapping: Dict[str, Optional[str]]):
    date_col = mapping.get("date")
    amount_col = mapping.get("amount")
    if not date_col or date_col not in df.columns:
        st.info("Δεν βρέθηκε στήλη ημερομηνίας.")
        return

    df_dt = coerce_datetime(df, date_col).dropna(subset=[date_col])
    if df_dt.empty:
        st.info("Δεν υπάρχουν έγκυρες ημερομηνίες για προβολή.")
        return

    df_dt = df_dt.copy()
    df_dt["__month"] = df_dt[date_col].dt.to_period("M").dt.to_timestamp()
    if amount_col and amount_col in df_dt.columns and np.issubdtype(df_dt[amount_col].dtype, np.number):
        series = df_dt.groupby("__month")[amount_col].sum().reset_index()
        fig = px.area(series, x="__month", y=amount_col, title="Χρονοσειρά ποσού ανά μήνα")
    else:
        series = df_dt.groupby("__month").size().reset_index(name="count")
        fig = px.area(series, x="__month", y="count", title="Χρονοσειρά πλήθους ανά μήνα")
    fig.update_traces(mode="lines")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Στοιχεία Συμβάσεων – Dashboard", layout="wide")
    st.title("Στοιχεία Συμβάσεων – Dashboard")

    st.sidebar.header("Ρυθμίσεις δεδομένων")
    default_xlsx_exists = Path(DEFAULT_XLSX_PATH).exists()
    default_csv_exists = Path(DEFAULT_CSV_DIR).exists()
    data_mode = st.sidebar.radio(
        "Πηγή δεδομένων",
        options=("CSV (cleaned)", "Excel (.xlsx)"),
        index=0 if default_csv_exists else 1,
        help="Χρησιμοποιήστε καθαρά CSV ή το αρχικό Excel.")

    df: Optional[pd.DataFrame] = None

    if data_mode == "CSV (cleaned)":
        st.sidebar.caption(f"Προεπιλογή φακέλου: {DEFAULT_CSV_DIR}")
        csv_df: Optional[pd.DataFrame] = None
        if default_csv_exists:
            csv_files = list_csv_files(DEFAULT_CSV_DIR)
            if csv_files:
                selected_csv = st.sidebar.selectbox("CSV αρχείο", csv_files, index=0, format_func=lambda p: Path(p).name)
                if selected_csv:
                    csv_df = load_csv_dataframe(selected_csv)
        uploaded_csv = st.sidebar.file_uploader("Ή ανεβάστε CSV", type=["csv"]) 
        if uploaded_csv is not None:
            csv_df = load_csv_dataframe(uploaded_csv)
        if csv_df is None:
            st.info("Επιλέξτε ένα CSV από τον φάκελο ή ανεβάστε CSV.")
            return
        df = csv_df
        drop_empty = False  # CSVs are already cleaned by exporter
    else:
        selected_sheet: Optional[str] = None
        source_ref: Optional[Union[io.BytesIO, str, Path]] = None
        excel_choice = st.sidebar.radio(
            "Πηγή αρχείου Excel",
            options=("Default path", "Upload"),
            index=0 if default_xlsx_exists else 1,
        )
        if excel_choice == "Default path" and default_xlsx_exists:
            try:
                source_ref = DEFAULT_XLSX_PATH
                sheets = list_sheet_names(source_ref)
                if sheets:
                    selected_sheet = st.sidebar.selectbox("Sheet", sheets, index=0)
            except Exception as e:
                st.sidebar.error(f"Σφάλμα φόρτωσης αρχείου: {e}")
        else:
            uploaded = st.sidebar.file_uploader("Ανεβάστε αρχείο .xlsx", type=["xlsx"]) 
            if uploaded is not None:
                try:
                    source_ref = uploaded
                    sheets = list_sheet_names(source_ref)
                    if sheets:
                        selected_sheet = st.sidebar.selectbox("Sheet", sheets, index=0)
                except Exception as e:
                    st.sidebar.error(f"Σφάλμα φόρτωσης αρχείου: {e}")

        if source_ref is None or selected_sheet is None:
            st.info("Φορτώστε ένα αρχείο Excel για να ξεκινήσετε.")
            return

        # Preview without headers to pick header row/depth
        with st.expander("Προεπισκόπηση (χωρίς επικεφαλίδες)", expanded=False):
            preview_df = load_excel_preview(source_ref, sheet_name=selected_sheet, nrows=20)
            st.dataframe(preview_df, use_container_width=True)

        st.sidebar.header("Επικεφαλίδες (headers)")
        suggested_header = _suggest_header_row_index_1based(preview_df)
        header_row_1based = st.sidebar.number_input("Γραμμή επικεφαλίδων (1-based)", min_value=1, max_value=max(1, len(preview_df)), value=suggested_header, step=1)
        header_depth = st.sidebar.slider("Βάθος επικεφαλίδων (rows)", min_value=1, max_value=3, value=1)
        drop_empty = st.sidebar.checkbox("Αφαίρεση κενών γραμμών/στηλών", value=True)

        # Load with chosen header settings
        df = load_excel_dataframe(
            source_ref,
            sheet_name=selected_sheet,
            header_start_row_index_1based=int(header_row_1based),
            header_depth=int(header_depth),
        )

        # Clean/flatten columns
        df = df.copy()
        df.columns = _flatten_and_clean_columns(df.columns)
        if drop_empty:
            # Drop columns completely empty and rows completely empty
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="all")

    # Column mapping (hidden by default; show only for admins)
    inferred = infer_column_mapping(df)
    admin_mode = st.sidebar.checkbox("Λειτουργία διαχειριστή", value=False, help="Εμφάνιση ρυθμίσεων αντιστοίχισης στηλών.")

    def select_col(label: str, default_value: Optional[str]) -> Optional[str]:
        options = [None] + list(df.columns)
        index = options.index(default_value) if default_value in options else 0
        return st.sidebar.selectbox(label, options=options, index=index, format_func=lambda x: x if x is not None else "(None)")

    if admin_mode:
        st.sidebar.header("Αντιστοίχιση στηλών")
        date_col = select_col("Ημερομηνία", inferred.get("date"))
        amount_col = select_col("Ποσό", inferred.get("amount"))
        customer_col = select_col("Πελάτης", inferred.get("customer"))
        contractor_col = select_col("Ανάδοχος/Προμηθευτής", inferred.get("contractor"))
        status_col = select_col("Κατάσταση", inferred.get("status"))
        payment_date_col = select_col("Ημερομηνία Πληρωμής (optional)", inferred.get("payment_date"))
        payment_amount_col = select_col("Ποσό Πληρωμής (optional)", inferred.get("payment_amount"))
        category_col = select_col("Κατηγορία (optional)", inferred.get("category"))
    else:
        date_col = inferred.get("date")
        amount_col = inferred.get("amount")
        customer_col = inferred.get("customer")
        contractor_col = inferred.get("contractor")
        status_col = inferred.get("status")
        payment_date_col = inferred.get("payment_date")
        payment_amount_col = inferred.get("payment_amount")
        category_col = inferred.get("category")

    mapping = {
        "date": date_col,
        "amount": amount_col,
        "customer": customer_col,
        "contractor": contractor_col,
        "status": status_col,
        "payment_date": payment_date_col,
        "payment_amount": payment_amount_col,
        "category": category_col,
    }

    # Tabs for dashboards (limited freedom per your spec)
    tab_overall, tab_payments, tab_contractor = st.tabs([
        "Overall", "Payments", "Per Contractor"
    ])

    with tab_overall:
        # Simple, clear filter inside the tab (contract date)
        df_use = df
        date_col = mapping.get("date")
        if date_col and date_col in df_use.columns:
            df_use = coerce_datetime(df_use, date_col)
            valid_dates = df_use[date_col].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                st.caption("Φίλτρο: Ημερομηνία σύμβασης")
                start_date, end_date = st.date_input("Εύρος", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                if isinstance(start_date, tuple):
                    start_date, end_date = start_date
                mask = (df_use[date_col] >= pd.to_datetime(start_date)) & (df_use[date_col] <= pd.to_datetime(end_date))
                df_use = df_use.loc[mask]
        render_overall_tab(df_use, mapping)
        st.download_button("Λήψη CSV (Overall)", data=df_use.to_csv(index=False).encode("utf-8-sig"), file_name="overall.csv", mime="text/csv")

    with tab_payments:
        # Simple, clear filter inside the tab (payment date)
        df_use = df
        payment_date_col = mapping.get("payment_date")
        if payment_date_col and payment_date_col in df_use.columns:
            df_use = coerce_datetime(df_use, payment_date_col)
            valid_dates = df_use[payment_date_col].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                st.caption("Φίλτρο: Ημερομηνία πληρωμής")
                start_date, end_date = st.date_input("Εύρος ", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="paydate")
                if isinstance(start_date, tuple):
                    start_date, end_date = start_date
                mask = (df_use[payment_date_col] >= pd.to_datetime(start_date)) & (df_use[payment_date_col] <= pd.to_datetime(end_date))
                df_use = df_use.loc[mask]
        render_payments_tab(df_use, mapping)
        st.download_button("Λήψη CSV (Payments)", data=df_use.to_csv(index=False).encode("utf-8-sig"), file_name="payments.csv", mime="text/csv")

    with tab_contractor:
        render_contractor_tab(df, mapping)


if __name__ == "__main__":
    main()


