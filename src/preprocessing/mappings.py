'''Mapping dictionaries for preprocessing the data.'''
import pandas as pd
CONVENIO_MAPPING = {
    "UNIMED RECIFE": ["UR", "U RECIFE", "UNIM RECIFE", "UNIMED REC", "UNIMED RECIFE",
                      "UNIMED R", "UNIMED RE", "UN RECIFE", "URECIFE"],
    "UNIMED GENERICO": ["UNIMED", "UNIM"],
    "SAUDE BRADESCO": ["SAUDE BARDESCO", "BRADESCO", "SAUDE BRADESCO", "S BRADESCO", "SB", "S B"],
    "SAUDE RECIFE": ["S RECIFE", "SAUDE RECIFE", "SAUDE REC", "SAUDERECIFE", "SR", "SAUDE RE"],
    "SANTA HELENA": ["STA HE", "S HELEN", "SANT HELEN", "STA HEL", "SHELEN",
                     "STHELENA", "SANTA HELEN",
                     "STA HELENA", "S HELENA", "SHELENA", "SANT H", "SH",
                     "STAHER", "SANTA HELENA"],
    "NORCLINICAS": ["NORCLIN", "NORCLINICAS", "NORCLINICA", "NORCL", "NORC",
                    "NORLCINICAS", "NORCLI", "NORCINICAS", "NORCLINIC",
                    "NORCLINC", "NORCLINIA"],
    "MEDISERVICE": ["MEDISERVICE", "MEDISERV", "MEDISE", "MEDISERVIC",
                    "MERSEV", "MEDISERVI", "MEDISERVICE", "MEDISERVECE",
                    "MEDSERV", "MEDISER", "MEDIS", "MEDSER", "MEDESV", "MEDESE"],
    "MED": ["MED"],
    "REAL SAUDE": ["REAL S", "REAL", "REAL SAUDE", "S REAL", "REALS", "RELA SAUDE"],
    "IDEAL SAUDE": ["IDEAL", "IDEAL S", "IDEAL SAUDE", "ODEAL S"],
    "MEDIAL": ["MEDIAL", "MEDIAL S", "MEDIAL SAUDE", "MEDIA S"],
    "FACHESF": ["FACHESF", "FACHES", "FACHESP", "FACHAESF", "FACJESF"],
    "SEMEPE": ["SEMEPE", "SEMEP"],
    "GRUPO SAUDE": ["GRUPO", "GRUPO SAUDE", "GS"],
    "PARTICULAR": ["PARTICULAR", "PARTICULRA", "PART", "PARTICL", "PARTIC", "P", "PATRT"],
    "CAMED": ["CAMED", "ACMED"],
    "CIRCULO CORACAO": ["C CORACAO", "CIRCULO COR", "CIRCULO", "CIRCC", "C COR", "CC"],
    "SAUDE EXCELSIOR": ["EXCELSIOR", "SAUDE EXCELSIOR", "EXCELSIOR RHP", "SEXCEL", "SE"],
    "PETROBRAS": ["PETROB", "PETROBRAS", "PETROF", "PETROFLEX", "PETROX", "PETRO"],
    "AERONAUTICA": ["AERONAUTICA", "AERON", "AERO"],
    "UNIBANCO": ["UNIBANCO", "UNIBAN", "UNIB"],
    "SAUDE CAIXA": ["S CX", "S CAIXA", "SAUDE CAIXA", "CAIXA"],
    "BLUE LIFE": ["BLUL", "BLUELI", "BLUE LIFE", "BL", "BLUELIFE"],
    "AMEPE CAMPE": ["AMEP CAMP", "AMEPCA", "AMEPE CAMPE", "AMEPE", "AMEP", "CAMPE"],
    "ASSEFAZ": ["ASSEFA", "ASSEF", "ASSEFAZ"],
    "PLAN ASSISTE": ["PLANSF", "PLAN ASSIST", "ASSIST", "PLAN A",
                     "PLAN ASSISTE", "PLAN", "PLAN ASS"],
    "CORREIOS": ["CORR", "COREIOS", "CORREIO", "CORREIOS"],
    "GAMA": ["GAMA S", "GAMA SAUDE", "GAMA"],
    "GOLDEN CROSS": ["GOLDEN", "GOLD", "GOLDEN CROSS", "GC"],
    "NOTREDAME": ["NOTREDAME", "NOTRE DAME"],
    "CORTESIA": ["CORT", "CORTESIA"],
    "REEMBOLSO SAUDE": ["RE", "REE SA", "REEMB"],
    "COMSAUDE": ["COM SAUDE", "COMSAUDE"],
    "OAB": ["OAB", "OAB SAUDE"],
    "BANCO CENTRAL": ["BANCO CENTRAL", "BCO CE"],
    "SANTA LUCIA": ["SANTA LUCIA", "STA LUCIA"],
    "SANTA CLARA": ["SANTA CLARA", "STA CLARA"],
    "EMBRATEL": ["EMBRATEL", "EMBRAT"],
    "FISCO SAUDE": ["FISCO", "FISCOS", "FISCO SAUDE", "FISCO S", "FISCOSAUDE"],
    "SUL AMERICA": ["SULA", "SUL AMERICA", "SULAM", "S AMERICA", "SUL AM",
                    "SULAMERICA", "S AM", "SUL AMER", "S AMER", "SUL-AMER",
                    "SULAMER", "SA", "S A"],
    "AMERICA SAUDE": ["AMERICA SAUDE", "AMERIACA S", "AMERICA", "AMERICA S", "AS"],
    "CASSI": ["CASSI", "CASS"],
    "TABELA GENERICO": ["TABELA", "TAB", "TABEL", "TBA"],
    "VIVA": ["VIVA", "VIVO"],
    "UNIMED GUARARAPES": ["U GUARARAPE", "U GUARAP", "UNIMED GUARARAPES",
                          "U GUARARAP", "U GUARA", "U GUARARAPES",
                          "UNIMED GUA", "UG", "U G"],
    "AGF": ["AGF", "AGF S"],
    "FUNCEF": ["FUNCEF", "FUNSEF"],
    "SUS": ["SUS", "SUS TMO"],
    "UNIMED OUTROS": ["UNIMED FORATLEZA", "UCARUARU", "UR MACEIO", "UNIMED PETROLINA"],
    "OUTROS": ["AMB", "BLSERV", "BANCEM", "SBPREF", "ALIANS",
        "SAQ", "CASSI + TAB", "CASSI PART", "HOSPITAL RESIDENCIA",
        "JORGE DE MEDEIROS", "BS", "MERIDIONAL", "SUDENE", "P UCMF", "PRET",
        "FIOPREV", "CISAM", "PM", "EXERCITO", "EMBRA", "AMIL", "TAB SB",
        "TAB UR"]
}

CONVENIO_LOOKUP = {variant: canonical for canonical, variants in CONVENIO_MAPPING.items(
    ) for variant in variants}

ANOMALY_MAPPING = {
    "NORMAL": ["NORMAL", "NORMAIS"]
}

ANOMALY_LOOKUP = {variant: canonical for canonical, variants in ANOMALY_MAPPING.items(
    ) for variant in variants}

SEX_MAPPING = {
    "M": ["M", "MASCULINO"],
    "F": ["F", "FEMININO"]
}

SEX_LOOKUP = {variant: canonical for canonical, variants in SEX_MAPPING.items(
    ) for variant in variants}

DN_TO_AGE_MAPPING = {
    "8anos/e/3/m": 8.25,
    "4/anos": 4.0,
    "2/anos": 2.0,
    "1a/7/m": 1.58,
    "5/anos": 5.0,
    "3anos": 3.0,
    "16a": 16.0,
    "52/anos": 52.0,
    "3a": 3.0,
    "4a": 4.0,
    "11/anos": 11.0,
    "10m": 0.83,
    "8/anos": 8.0,
    "9a": 9.0,
    "1a/7m": 1.58,
    "1a2m": 1.17,
    "4A": 4.0
}

DN_MANUAL_FIXES = {
    "25/06/996": "25/06/1996",
    "26/010/99": "26/10/1999",
    "10/06/97arritmia": "10/06/1997",
    "07/05/096": "07/05/1996",
    "11/0503": "11/05/2003",
    "25/06/009": "25/06/2009",
    "04/0897": "04/08/1997",
    "20/052003": "20/05/2003",
    "04/02/094": "04/02/1994",
    "30/0898": "30/08/1998",
    "14/11/990": "14/11/1990",
    "15/11/996": "15/11/1996",
    "30/09/997": "30/09/1997",
    "17/005/00": "17/05/2000",
    "01/04/992": "01/04/1992",
    "22/012/00": "22/12/2000",
    "19/04/096": "19/04/1996",
    "28/03/0/5": "28/03/2005",
    "05/10/994": "05/10/1994",
    "25/06/993": "25/06/1993",
    "03/07/02004": "03/07/2004",
    "29/10/995": "29/10/1995",
    "07/0303": "07/03/2003",
    "0912/08": "09/12/2008",
    "021/08/2000": "21/08/2000",
    "14/012/05": "14/12/2005",
    "01/011/05": "01/11/2005",
    "09/12/993": "09/12/1993",
    "07/01/099": "07/01/1999",
    "Ago/28,/1991": "28/08/1991",
    "26/02/097": "26/02/1997",
    "25/05/995": "25/05/1995",
    "!2/01/97": "12/01/1997",
    "11/11/992": "11/11/1992",
    "17/12/005": "17/12/2005",
    "13/07/997": "13/07/1997",
    "01/02/999": "01/02/1999",
    "16/03/003": "16/03/2003",
    "24/08/992": "24/08/1992",
    "13/01/098": "13/01/1998",
    "10/06/004": "10/06/2004",
    "17/05/095": "17/05/1995",
    "17/0899": "17/08/1999",
    "23/09/003": "23/09/2003",
    "26/010/2004": "26/10/2004",
    "23/02/061": "23/02/1961",
    "21/091997": "21/09/1997",
    "04/01/067": "04/01/1967",
    "05/05/998": "05/05/1998",
    "07/03/002": "07/03/2002",
    "17/12/02CIV/2mm": "17/12/2002",
    "10/102004": "10/10/2004",
    "02/12/096": "02/12/1996",
    "27/03/998": "27/03/1998",
    "14/0596": "14/05/1996",
    "04/12/095": "04/12/1995",
    "08/0599": "08/05/1999",
    "09/02/998": "09/02/1998",
    "20/01/994": "20/01/1994",
    "04/10/097": "04/10/1997",
    "08/10/996": "08/10/1996",
    "21/03/988": "21/03/1988",
    "01/09/992": "01/09/1992",
    "12/03/998": "12/03/1998",
    "1902/2006": "19/02/2006",
    "03/11/991": "03/11/1991",
    "13/0599": "13/05/1999",
    "24/05/996": "24/05/1996",
    "05/07/097": "05/07/1997",
    "30/0797": "30/07/1997"
}

AGE_BINS = [0.0, 0.25,1.0,3.0,7.0,12.0,18.0,20.0]
AGE_LABELS = ["0-3m", "3-12m", "1-3a", "3-7a", "7-12a", "12-18a", "18-20a"]

BOUNDS_BY_AGE = {
    "peso": {
        "0-3m": (0.4, 12.0),
        "3-12m": (2.0, 22.0),
        "1-3a": (3.5, 40.0),
        "3-7a": (6.0, 80.0),
        "7-12a": (10.0, 150.0),
        "12-18a": (15.0, 230.0),
        "18-20a": (18.0, 250.0),
        "unknown": (0.4, 250.0), # caso a idade seja NaN
    },
    "altura": {
        "0-3m": (25.0, 75.0),
        "3-12m": (30.0, 100.0),
        "1-3a": (36.0, 135.0),
        "3-7a": (42.0, 180.0),
        "7-12a": (48.0, 225.0),
        "12-18a": (54.0, 265.0),
        "18-20a": (54.0, 273.0),
        "unknown": (25.0, 273.0), # caso a idade seja NaN
    },
    "pa_sistolica": {
        "0-3m": (20.0, 180.0),
        "3-12m": (20.0, 250.0),
        "1-3a": (20.0, 300.0),
        "3-7a": (20.0, 300.0),
        "7-12a": (20.0, 300.0),
        "12-18a": (20.0, 300.0),
        "18-20a": (20.0, 300.0),
        "unknown": (20.0, 300.0), # caso a idade seja NaN
    },
    "pa_diastolica": {
        "0-3m": (10.0, 140.0),
        "3-12m": (10.0, 180.0),
        "1-3a": (10.0, 200.0),
        "3-7a": (10.0, 200.0),
        "7-12a": (10.0, 200.0),
        "12-18a": (10.0, 200.0),
        "18-20a": (10.0, 200.0),
        "unknown": (10.0, 200.0), # caso a idade seja NaN
    },
    "fc": {
        "0-3m": (20.0, 320.0),
        "3-12m": (20.0, 320.0),
        "1-3a": (20.0, 300.0),
        "3-7a": (20.0, 300.0),
        "7-12a": (15.0, 300.0),
        "12-18a": (15.0, 300.0),
        "18-20a": (15.0, 300.0),
        "unknown": (15.0, 320.0), # caso a idade seja NaN
    }
}


def get_bounds_by_age(age_series, bounds_dict):
    '''Get the lower and upper bounds for a given age series and variable.
    can be: "peso" or "altura"'''
    age_groups = (
        pd.cut(age_series, bins=AGE_BINS, labels=AGE_LABELS, right=False)
        .astype(object)
        .fillna("unknown")
    )
    lowers = age_groups.map({k: v[0] for k, v in bounds_dict.items()}).astype(float)
    uppers = age_groups.map({k: v[1] for k, v in bounds_dict.items()}).astype(float)
    return lowers, uppers
