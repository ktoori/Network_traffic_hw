from __future__ import annotations

import re

ENCODING_UTF8_SIG = "utf-8-sig"

COL_EXP_MONTHS = "Стаж (месяцев)"
COL_IT_DIRECTION = "IT_направление"

LEVEL_COL = "level"
TITLE_TEXT_COL = "__title_text__"

TITLE_PREFIX_LAST = "Последняя должность_"
TITLE_PREFIX_WANTED = "Ищет работу на должность_"
WORKPLACE_PREFIX = "Последнее место работы_"

TITLE_PREFIXES = (TITLE_PREFIX_LAST, TITLE_PREFIX_WANTED)

DROP_PREFIXES_LEAKY_OR_NOISY = (
    TITLE_PREFIX_LAST,
    TITLE_PREFIX_WANTED,
    WORKPLACE_PREFIX,
)

CSV_X_CLEAN = "X_clean.csv"
CSV_Y_LEVEL = "y_level.csv"
CSV_DF_WITH_LEVEL = "df_with_level.csv"

PLOT_BAR = "class_balance_bar.png"
PLOT_PIE = "class_balance_pie.png"
BALANCE_TXT = "class_balance.txt"

SENIOR_PATTERN = (
    r"(ведущ|старш|руковод|директор|head|lead|principal|architect|team\s*lead|tech\s*lead)"
)
JUNIOR_PATTERN = r"(младш|junior|intern|trainee|стажер|начинающ)"

SENIOR_TITLE_RE = re.compile(SENIOR_PATTERN, flags=re.IGNORECASE)
JUNIOR_TITLE_RE = re.compile(JUNIOR_PATTERN, flags=re.IGNORECASE)

LEVEL_JUNIOR = "junior"
LEVEL_MIDDLE = "middle"
LEVEL_SENIOR = "senior"
