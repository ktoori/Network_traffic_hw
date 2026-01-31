"""
Configuration and constants for the preprocessing pipeline hh.csv.
"""

from typing import Dict, List, Optional

# Target column name
TARGET_COLUMN: str = "ЗП"

# Currency conversion rates to RUB
CURRENCY_RATES: Dict[str, float] = {
    "rub": 1.0,
    "руб": 1.0,
    "rur": 1.0,
    "usd": 90.0,
    "eur": 98.0,
    "uah": 2.4,
    "грн": 2.4,
    "kzt": 0.2,
    "kzs": 0.2,
    "kgs": 1.0,
    "byn": 28.0,
    "бел": 28.0,
    "azn": 53.0,
}

# Mapping for City translation and normalization
CITY_TRANSLATION_MAP: Dict[str, Optional[str]] = {
    "moscow": "Москва",
    "saint petersburg": "Санкт-Петербург",
    "st petersburg": "Санкт-Петербург",
    "remote": None,
}

# Gender normalization map
GENDER_MAP: Dict[str, str] = {
    "male": "Мужчина",
    "female": "Женщина",
    "мужчина": "Мужчина",
    "женщина": "Женщина",
}

# Keywords for 'Employment' (Занятость) one-hot encoding
EMPLOYMENT_PATTERNS: Dict[str, List[str]] = {
    "полная": ["полная занятость", "full time"],
    "частичная": ["частичная занятость", "part time"],
    "проект": ["проектная работа", "project work"],
    "стажировка": ["стажировка", "work placement"],
    "волонтерство": ["волонтерство", "volunteering"],
}

# Keywords for 'Schedule' (График) one-hot encoding
SCHEDULE_PATTERNS: Dict[str, List[str]] = {
    "Полный_день": ["полный день", "full day"],
    "Гибкий_график": ["гибкий график", "flexible schedule"],
    "Удаленная": ["удаленная работа", "remote working"],
    "Сменный_график": ["сменный график", "shift schedule"],
    "Вахтовый": ["вахтовый метод", "rotation based work"],
}

# Keywords for Education classification
EDUCATION_TECHNICAL_KEYWORDS: List[str] = [
    "техническ", "радио", "электрон", "авиацион",
    "телекоммуникац", "информатик", "кибернет",
]

EDUCATION_HUMANITARIAN_KEYWORDS: List[str] = [
    "гуманитар", "юрид", "прав", "эконом",
]

# Columns to reduce cardinality (keep top N)
COLS_TO_REDUCE_CARDINALITY: List[str] = [
    "Ищет работу на должность",
    "Последнее место работы",
    "Последняя должность",
    "Город",
]

CARDINALITY_LIMIT: int = 50

MIN_SALARY_RUB = 9000
MAX_SALARY_RUB = 1_500_000

Y_WINSORIZE_ENABLED = True
Y_WINSORIZE_LO_Q = 0.01
Y_WINSORIZE_HI_Q = 0.99

AGE_CLIP_ENABLED = True
AGE_MIN = 14
AGE_MAX = 80

EXP_MONTHS_CLIP_ENABLED = True
EXP_MONTHS_MAX = 600
