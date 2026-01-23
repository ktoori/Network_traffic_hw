"""
Chain of Responsibility handlers for hh.csv preprocessing.
"""

from typing import Any, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import re

logger = logging.getLogger(__name__)

CURRENCY_RATES_TO_RUB = {
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



class Handler:
    """
    Abstract base class for Chain of Responsibility handlers.
    """

    def __init__(self) -> None:
        self._next: Optional["Handler"] = None

    def set_next(self, handler: "Handler") -> "Handler":
        """
        Set the next handler in the chain.

        Parameters
        ----------
        handler : Handler
            Next handler to be executed.

        Returns
        -------
        Handler
            The same handler.
        """
        self._next = handler
        return handler

    def handle(self, data: Any) -> Any:
        """
        Process input data.

        Must be overridden in subclasses.

        Parameters
        ----------
        data : Any
            Input data for processing.

        Returns
        -------
        Any
            Processed data.
        """
        raise NotImplementedError("handle() must be implemented in subclasses")

    def _call_next(self, data: Any) -> Any:
        """
        Pass data to the next handler if it exists.
        """
        if self._next is None:
            return data
        return self._next.handle(data)


class CSVLoadHandler(Handler):
    """
    Load CSV file into a pandas DataFrame.
    """

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Read CSV file by path.

        Parameters
        ----------
        data : Any
            Path to CSV file.

        Returns
        -------
        pandas.DataFrame
            Loaded DataFrame.
        """
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # utf-8-sig is used to correctly handle BOM if present
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, encoding="utf-8-sig")

        # Store source path for output files
        df.attrs["__source_path__"] = str(path)
        logger.info("CSV loaded: %s", path)

        print("Размер:", df.shape)
        print("\nСтолбцы:", df.columns.tolist())
        print("\nПервые строки:\n", df.head())
        '''print(df['Unnamed: 0'].unique())
        print(df['Пол, возраст'].unique())
        print(df['ЗП'].unique())
        print(df['Ищет работу на должность:'].unique())
        print(df['Город'].unique())
        print(df['Занятость'].unique())
        print(df['График'].unique())
        print(df['Опыт (двойное нажатие для полной версии)'].unique())
        print(df['Последенее/нынешнее место работы'].unique())
        print(df['Последеняя/нынешняя должность'].unique())
        print(df['Образование и ВУЗ'].unique())
        print(df['Обновление резюме'].unique())
        print(df['Авто'].unique())'''
        return self._call_next(df)


class CleanTextHandler(Handler):
    """
    Clean textual data from unwanted unicode and control characters.
    """

    @staticmethod
    def _clean_value(value: Any) -> Any:
        """
        Clean a single cell value if it is a string.
        """
        if pd.isna(value) or not isinstance(value, str):
            return value

        # Remove BOM and non-breaking space
        value = value.replace("\ufeff", "").replace("\xa0", " ")

        # Remove control characters except common whitespace
        cleaned = "".join(
            character for character in value if character >= " " or character in ("\t", "\n", "\r")
        )
        return cleaned.strip()

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Apply text cleaning to all object-type columns.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CleanTextHandler expects pandas.DataFrame")

        df = data.copy()
        text_columns = df.select_dtypes(include=[object]).columns

        for col in text_columns:
            df[col] = df[col].map(self._clean_value)

        logger.info("Text columns cleaned.")
        return self._call_next(df)


class FillMissingHandler(Handler):
    """
    Handle missing values using a fixed target column.
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        self._target = target

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Remove rows with missing target values and fill other missing values.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("FillMissingHandler expects pandas.DataFrame")

        if self._target not in data.columns:
            raise ValueError(f"Target column '{self._target}' not found")

        df = data.copy()

        # Drop rows with missing target values
        df = df[df[self._target].notna()].reset_index(drop=True)

        # Fill numeric columns with mean values
        for col in df.select_dtypes(include=["number"]):
            df[col] = df[col].fillna(df[col].mean())

        # Fill categorical columns with placeholder value
        for col in df.select_dtypes(include=[object]):
            df[col] = df[col].fillna("Не_указано")

        df.attrs["__target__"] = self._target
        logger.info("Missing values processed.")
        return self._call_next(df)


class AdvancedParsingHandler(Handler):
    """
    Perform complex parsing and feature engineering on raw string columns.
    Handles gender/age, salary, city details, employment type, schedule,
    experience calculation, education parsing, and cardinality reduction.
    """

    def _parse_gender_age(self, row: pd.Series) -> pd.Series:
        """
        Parse 'Пол, возраст' column.
        Format: 'Мужчина , 40 лет , родился 14 июня 1999'
        Returns: Gender, Age (int), Birth Date (datetime)
        """
        value = row["Пол, возраст"]
        if pd.isna(value):
            return pd.Series([None, None, None])

        parts = [p.strip() for p in value.split(",")]
        if len(parts) < 3:
            return pd.Series([None, None, None])

        # 1. Gender
        gender = parts[0]

        # 2. Age
        age_match = re.search(r"\d+", parts[1])
        age = int(age_match.group()) if age_match else None

        # 3. Birth Date
        dob_str = parts[2].replace("родился ", "").replace("родилась ", "").strip()
        months_map = {
            "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
            "мая": "05", "июня": "06", "июля": "07", "августа": "08",
            "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"
        }

        for ru_month, num_month in months_map.items():
            if ru_month in dob_str:
                dob_str = dob_str.replace(ru_month, num_month)
                break

        try:
            birth_date = pd.to_datetime(dob_str, format="%d %m %Y", errors='coerce')
        except Exception:
            birth_date = None

        return pd.Series([gender, age, birth_date])

    def _parse_experience(self, value: str) -> int:
        """
        Parse work experience into total months.
        Returns total months as int.
        """
        if pd.isna(value):
            return 0

        text = str(value).lower()

        if "не указано" in text or "нет опыта" in text:
            return 0

        # take only first line / header
        header = text.split("\n")[0]

        years = 0
        months = 0

        # russian
        y_match = re.search(r"(\d+)\s*(?:год|года|лет)", header)
        m_match = re.search(r"(\d+)\s*(?:месяц|месяца|месяцев)", header)

        # english fallback
        if not y_match:
            y_match = re.search(r"(\d+)\s*(?:year|years)", header)
        if not m_match:
            m_match = re.search(r"(\d+)\s*(?:month|months)", header)

        if y_match:
            years = int(y_match.group(1))
        if m_match:
            months = int(m_match.group(1))

        # safety cap: 80 years max
        total_months = years * 12 + months
        return min(total_months, 80 * 12)

    def _parse_city(self, value: str) -> Tuple[str, bool, bool]:
        """
        Parse 'Город'.
        Returns: city (str), relocation (bool), business_trips (bool)
        """
        CITY_TRANSLATION_MAP = {
            "moscow": "Москва",
            "saint petersburg": "Санкт-Петербург",
            "st petersburg": "Санкт-Петербург",
            "remote": None,
        }
        value_lower = value.lower()

        for eng, rus in CITY_TRANSLATION_MAP.items():
            if eng in value_lower:
                value = rus if rus is not None else value

        if pd.isna(value):
            return None, False, False

        # remove everything in parentheses
        clean_value = re.sub(r"\(.*?\)", "", value)

        parts = [p.strip() for p in clean_value.split(",") if p.strip()]

        city_parts = []
        relocation = False
        trips = False

        for part in parts:
            part_lower = part.lower()

            # relocation logic
            if "хочу переехать" in part_lower:
                relocation = True
            elif "переезду" in part_lower:
                if "не готов" not in part_lower:
                    relocation = True

            # business trips logic
            elif "командировкам" in part_lower:
                if "не готов" not in part_lower:
                    trips = True

            # city name
            else:
                city_parts.append(part)

        city = ", ".join(city_parts) if city_parts else None
        return city, relocation, trips

    def _parse_education(self, value: str) -> Tuple[str, int, str]:
        """
        Parse 'Образование'.
        Format: 'Высшее образование 1999 СОЧИНСКИЙ...'
        Returns: Level, Year (int), Institution
        """
        if pd.isna(value):
            return None, None, None

        match = re.search(r"(.*?образование)\s+(\d{4})\s+(.*)", value)
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
        return None, None, None

    def _reduce_cardinality(self, series: pd.Series, top_n: int = 30) -> pd.Series:
        """
        Keep only top_n most frequent values, replace others with 'другое'.
        """
        top_values = series.value_counts().head(top_n).index
        return series.apply(lambda x: x if x in top_values else "другое")

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Apply parsing and feature engineering steps.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("AdvancedParsingHandler expects pandas.DataFrame")

        df = data.copy()

        df = df.rename(columns={"Unnamed: 0": "ID"})
        df = df.drop(columns=["Обновление резюме"])

        # 1. Parse 'Пол, возраст'
        if "Пол, возраст" in df.columns:
            df[["Пол", "Возраст", "Дата рождения"]] = df.apply(self._parse_gender_age, axis=1)
            df = df.drop(columns=["Пол, возраст"])

        if "Пол" in df.columns:
            gender_map = {
                "male": "Мужчина",
                "female": "Женщина",
                "мужчина": "Мужчина",
                "женщина": "Женщина",
            }

            df["Пол"] = (
                df["Пол"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(gender_map)
            )

        # 2. Parse 'ЗП'
        if "ЗП" in df.columns:
            def parse_salary(cell):
                if pd.isna(cell):
                    return None

                text = str(cell).lower()

                # extract number
                num_match = re.search(r"(\d[\d\s]*)", text)
                if not num_match:
                    return None

                amount = float(num_match.group(1).replace(" ", ""))

                # detect currency
                rate = 1.0
                for key, val in CURRENCY_RATES_TO_RUB.items():
                    if key in text:
                        rate = val
                        break

                return amount * rate

            df["ЗП_РУБ"] = df["ЗП"].apply(parse_salary)
            df["ЗП_РУБ"] = pd.to_numeric(df["ЗП_РУБ"], errors="coerce")

            df = df.drop(columns=["ЗП"])

        df["ЗП_РУБ"] = df["ЗП_РУБ"].round(0).astype("Int64")
        df = df.rename(columns={"ЗП_РУБ": "ЗП"})

        # 3. Parse 'Город'
        if "Город" in df.columns:
            city_data = df["Город"].apply(self._parse_city).tolist()
            df["Город"] = [x[0] for x in city_data]
            df["Готовность к переезду"] = [x[1] for x in city_data]
            df["Готовность к командировкам"] = [x[2] for x in city_data]

        if "Город" in df.columns:
            df["Город"] = (
                df["Город"]
                .astype(str)
                .str.split(",")
                .str[0]
                .str.strip()
            )

        # 4. Parse 'Занятость' (One-hot logic)
        # Parse 'Занятость'
        if "Занятость" in df.columns:
            employment_map = {
                "полная": [
                    "полная занятость",
                    "full time",
                ],
                "частичная": [
                    "частичная занятость",
                    "part time",
                ],
                "проект": [
                    "проектная работа",
                    "project work",
                ],
                "стажировка": [
                    "стажировка",
                    "work placement",
                ],
                "волонтерство": [
                    "волонтерство",
                    "volunteering",
                ],
            }

            src = df["Занятость"].fillna("").str.lower()

            for key, patterns in employment_map.items():
                df[f"Занятость_{key}"] = src.apply(
                    lambda x: any(p in x for p in patterns)
                )

            df = df.drop(columns=["Занятость"])

        # 5. Parse 'График' (One-hot logic)
        # Parse 'График'
        if "График" in df.columns:
            schedule_map = {
                "Полный_день": [
                    "полный день",
                    "full day",
                ],
                "Гибкий_график": [
                    "гибкий график",
                    "flexible schedule",
                ],
                "Удаленная": [
                    "удаленная работа",
                    "remote working",
                ],
                "Сменный_график": [
                    "сменный график",
                    "shift schedule",
                ],
                "Вахтовый": [
                    "вахтовый метод",
                    "rotation based work",
                ],
            }

            src = df["График"].fillna("").str.lower()

            for key, patterns in schedule_map.items():
                df[f"График_{key}"] = src.apply(
                    lambda x: any(p in x for p in patterns)
                )

            df = df.drop(columns=["График"])

        # 6. Parse 'Опыт работы' (Keep only months)
        if "Опыт (двойное нажатие для полной версии)" in df.columns:
            df["Стаж (месяцев)"] = (
                df["Опыт (двойное нажатие для полной версии)"]
                .apply(self._parse_experience)
                .astype("int32")
            )

            df = df.drop(columns=["Опыт (двойное нажатие для полной версии)"])

        # 7. Parse 'Образование' (or 'Образование и ВУЗ')
        if "Образование и ВУЗ" in df.columns:
            edu_parsed = df["Образование и ВУЗ"].apply(self._parse_education).tolist()
            df["Уровень образования"] = [x[0] for x in edu_parsed]
            df["Год окончания"] = [x[1] for x in edu_parsed]
            df["Учебное заведение"] = [x[2] for x in edu_parsed]
            df = df.drop(columns=["Образование и ВУЗ"])

        if "Уровень образования" in df.columns:
            df["Уровень образования"] = df["Уровень образования"].apply(
                lambda x: "Другое"
                if isinstance(x, str) and len(x) > 50
                else x
            )

        # 8. Reduce cardinality for high-cardinality text columns
        text_cols_to_reduce = [
            "Ищет работу на должность:",
            "Последенее/нынешнее место работы",
            "Последеняя/нынешняя должность",
            "Город"
        ]

        for col in text_cols_to_reduce:
            if col in df.columns:
                # Optional: normalize text first (lowercase, strip)
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = self._reduce_cardinality(df[col], top_n=50)

        if "Учебное заведение" in df.columns:
            src = df["Учебное заведение"].fillna("").str.lower()
            technical_keywords = [
                "техническ", "радио", "электрон", "авиацион",
                "телекоммуникац", "информатик", "кибернет",
            ]

            humanitarian_keywords = [
                "гуманитар", "юрид", "прав", "эконом",
            ]

            df.loc[src.str.contains("|".join(technical_keywords)), "Категория_образования"] = "Технический"
            df.loc[src.str.contains("|".join(humanitarian_keywords)), "Категория_образования"] = "Гуманитарный"

            df["Город_УЗ"] = None

            df.loc[src.str.contains("москва"), "Город_УЗ"] = "Москва"
            df.loc[src.str.contains("санкт-петербург|спб"), "Город_УЗ"] = "Санкт-Петербург"

            df["IT_направление"] = src.str.contains(
                "информатик|программ|вычислительн|компьютер|сети",
                regex=True
            )

            df["Безопасность_направление"] = src.str.contains(
                "безопасност|защита|крипто",
                regex=True
            )

        df = df.drop(columns=["Дата рождения"])
        df = df.drop(columns=["Год окончания"])
        df = df.drop(columns=["Учебное заведение"])

        logger.info("Advanced parsing and feature engineering completed.")

        print("Размер:", df.shape)
        print("\nСтолбцы:", df.columns.tolist())
        print("\nПервые строки:\n", df.head())
        print(df['ID'].unique())
        print(df['Ищет работу на должность:'].unique())
        print(df['Город'].unique())
        print(df['Последенее/нынешнее место работы'].unique())
        print(df['Последеняя/нынешняя должность'].unique())
        print(df['Авто'].unique())
        print(df['Пол'].unique())
        print(df['Возраст'].unique())
        print(df['ЗП'].unique())
        print(df['Готовность к переезду'].unique())
        print(df['Готовность к командировкам'].unique())
        print(df['Занятость_полная'].unique())
        print(df['Занятость_частичная'].unique())
        print(df['Занятость_проект'].unique())
        print(df['Занятость_стажировка'].unique())
        print(df['Занятость_волонтерство'].unique())
        print(df['График_Полный_день'].unique())
        print(df['График_Гибкий_график'].unique())
        print(df['График_Удаленная'].unique())
        print(df['График_Сменный_график'].unique())
        print(df['График_Вахтовый'].unique())
        print(df['Уровень образования'].unique())
        print(df['Категория_образования'].unique())
        print(df['Город_УЗ'].unique())
        print(df['IT_направление'].unique())
        print(df['Безопасность_направление'].unique())
        print(df['Стаж (месяцев)'].unique())


        return self._call_next(df)


class EncodeHandler(Handler):
    """
    Encode categorical features using one-hot encoding.
    """

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Convert categorical columns into numeric representation.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("EncodeHandler expects pandas.DataFrame")

        df = pd.get_dummies(data, drop_first=False)
        df.attrs.update(data.attrs)
        logger.info("Categorical features encoded.")
        print("Размер:", df.shape)
        print("\nСтолбцы:", df.columns.tolist())
        print("\nПервые строки:\n", df.head())
        return self._call_next(df)


class FeatureTargetSplitHandler(Handler):
    """
    Split dataset into feature matrix X and target vector y.
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        self._target = target

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Split DataFrame into X and y numpy arrays.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("FeatureTargetSplitHandler expects pandas.DataFrame")

        if self._target not in data.columns:
            raise ValueError(f"Target column '{self._target}' not found")

        X = data.drop(columns=[self._target]).to_numpy(dtype=float)
        y = data[self._target].to_numpy(dtype=float)

        #######################
        # --- Sanity checks before saving numpy arrays ---

        # 1. Shape consistency
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have different number of rows: X={X.shape[0]}, y={y.shape[0]}"
            )

        # 2. Empty features check
        if X.shape[1] == 0:
            raise ValueError("X has zero feature columns after preprocessing")

        # 3. NaN / Inf checks
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")

        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values")

        # 4. Variance check (catch constant or broken features)
        zero_variance_features = np.where(X.std(axis=0) == 0)[0]
        if len(zero_variance_features) > 0:
            logger.warning(
                "Found %d zero-variance features (example indices: %s)",
                len(zero_variance_features),
                zero_variance_features[:10],
            )

        # 5. Debug preview (safe, lightweight)
        logger.info(
            "X shape: %s, y shape: %s, X sample (first row, first 10 features): %s",
            X.shape,
            y.shape,
            X[0, :10],
        )
        ######################

        data.attrs["__X__"] = X
        data.attrs["__y__"] = y
        logger.info("Features and target split completed.")

        return self._call_next(data)


class SaveNpyHandler(Handler):
    """
    Save processed features and target to .npy files.
    """

    def handle(self, data: Any) -> Tuple[str, str]:
        """
        Save X and y arrays to disk.

        Returns
        -------
        Tuple[str, str]
            Paths to saved x_data.npy and y_data.npy files.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("SaveNpyHandler expects pandas.DataFrame")

        X = data.attrs.get("__X__")
        y = data.attrs.get("__y__")
        source = data.attrs.get("__source_path__")

        if X is None or y is None or source is None:
            raise ValueError("Missing required data for saving")

        source_path = Path(source)
        x_path = source_path.with_name("x_data.npy")
        y_path = source_path.with_name("y_data.npy")

        np.save(x_path, X)
        np.save(y_path, y)

        logger.info("Saved x_data.npy and y_data.npy")
        return str(x_path), str(y_path)
