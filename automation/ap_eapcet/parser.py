import io
import pandas as pd


# ============================================================
# AP EAPCET OFFICIAL EXCEL PARSER
# ============================================================

EXPECTED_COLUMNS = [
    "SNO",
    "INSTCODE",
    "NAME OF THE INSTITUTION",
    "TYPE",
    "INST_REG",
    "DIST",
    "PLACE",
    "COED",
    "AFFL.",
    "ESTD",
    "A_REG",
    "branch_code",
    "OC_BOYS",
    "OC_GIRLS",
    "SC_BOYS",
    "SC_GIRLS",
    "ST_BOYS",
    "ST_GIRLS",
    "BCA_BOYS",
    "BCA_GIRLS",
    "BCB_BOYS",
    "BCB_GIRLS",
    "BCC_BOYS",
    "BCC_GIRLS",
    "BCD_BOYS",
    "BCD_GIRLS",
    "BCE_BOYS",
    "BCE_GIRLS",
    "OC_EWS_BOYS",
    "OC_EWS_GIRLS",
    "COLLFEE"
]


# ============================================================
# READ OFFICIAL EXCEL
# ============================================================

def read_official_excel(excel_bytes):

    print()
    print("Reading official AP EAPCET Excel...")

    excel_file = io.BytesIO(
        excel_bytes
    )

    excel = pd.ExcelFile(
        excel_file,
        engine="xlrd"
    )

    print()
    print("Sheets found:")

    for sheet in excel.sheet_names:

        print(
            f"  - {sheet}"
        )

    # The official file currently contains
    # one sheet.
    sheet_name = excel.sheet_names[0]

    raw = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        header=None,
        engine="xlrd"
    )

    print()
    print(
        f"Raw rows: {len(raw)}"
    )

    print(
        f"Raw columns: {len(raw.columns)}"
    )

    return raw


# ============================================================
# FIND HEADER ROW
# ============================================================

def find_header_row(raw):

    for index, row in raw.iterrows():

        values = [
            str(value).strip()
            for value in row.tolist()
        ]

        if (
            "INSTCODE" in values
            and "NAME OF THE INSTITUTION" in values
            and "branch_code" in values
        ):

            print()
            print(
                f"Header row found: {index}"
            )

            return index

    raise ValueError(
        "Could not find AP EAPCET header row."
    )


# ============================================================
# BUILD DATAFRAME
# ============================================================

def build_dataframe(raw):

    header_row = find_header_row(
        raw
    )

    headers = [
        str(value).strip()
        for value in raw.iloc[
            header_row
        ].tolist()
    ]

    df = raw.iloc[
        header_row + 1:
    ].copy()

    df.columns = headers

    # Remove empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove empty rows
    df = df.dropna(
        how="all"
    )

    return df


# ============================================================
# NORMALIZE AP EAPCET DATA
# ============================================================

def normalize_ap_eapcet(df):

    required = [
        "INSTCODE",
        "NAME OF THE INSTITUTION",
        "PLACE",
        "branch_code",
        "OC_BOYS",
        "OC_GIRLS",
        "COLLFEE"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    result = pd.DataFrame()

    # --------------------------------------------------------
    # COLLEGE
    # --------------------------------------------------------

    result["College Name"] = (
        df["NAME OF THE INSTITUTION"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # BRANCH
    # --------------------------------------------------------

    result["Branch"] = (
        df["branch_code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    result["Location"] = (
        df["PLACE"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    result["State"] = (
        "Andhra Pradesh"
    )

    # --------------------------------------------------------
    # EXAM
    # --------------------------------------------------------

    result["Exam"] = (
        "EAMCET"
    )

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    result["Year"] = 2024

    # --------------------------------------------------------
    # COLLEGE CODE
    # --------------------------------------------------------

    result["College Code"] = (
        df["INSTCODE"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # GENERAL CUTOFF
    #
    # Use the better of OC boys/girls
    # (smaller rank = better cutoff).
    # --------------------------------------------------------

    boys = pd.to_numeric(
        df["OC_BOYS"],
        errors="coerce"
    )

    girls = pd.to_numeric(
        df["OC_GIRLS"],
        errors="coerce"
    )

    result["Cutoff Rank"] = (
        pd.concat(
            [boys, girls],
            axis=1
        )
        .min(axis=1)
    )

    # --------------------------------------------------------
    # EXAM FEE
    # --------------------------------------------------------

    result["Eamcet Fees"] = pd.to_numeric(
        df["COLLFEE"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # EMPTY FIELDS
    #
    # These require separate authoritative sources.
    # --------------------------------------------------------

    result["Rating"] = None

    result["Management Fees"] = None

    result["Hostel Fees"] = None

    result["Image"] = ""

    result["Latitude"] = None

    result["Longitude"] = None

    result["Website"] = ""

    # --------------------------------------------------------
    # REMOVE INVALID RECORDS
    # --------------------------------------------------------

    result = result[
        result["College Name"].str.len() > 0
    ]

    result = result[
        result["Branch"].str.len() > 0
    ]

    result = result[
        result["Cutoff Rank"].notna()
    ]

    result = result[
        result["Eamcet Fees"].notna()
    ]

    # --------------------------------------------------------
    # CLEAN TYPES
    # --------------------------------------------------------

    result["Cutoff Rank"] = (
        result["Cutoff Rank"]
        .astype(int)
    )

    result["Eamcet Fees"] = (
        result["Eamcet Fees"]
        .astype(float)
    )

    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    result = result.reset_index(
        drop=True
    )

    return result