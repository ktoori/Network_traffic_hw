"""
Chain of Responsibility handlers for data processing.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class Handler(ABC):
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
            The next handler to execute.

        Returns
        -------
        Handler
            The handler passed as an argument (for chaining).
        """
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, data: Any) -> Any:
        """
        Process the data. Must be implemented by subclasses.

        Parameters
        ----------
        data : Any
            Input data.

        Returns
        -------
        Any
            Processed data.
        """
        pass

    def _call_next(self, data: Any) -> Any:
        """
        Pass data to the next handler in the chain if it exists.

        Parameters
        ----------
        data : Any
            Data to be passed to the next handler.

        Returns
        -------
        Any
            Result of the next handler or the original data if no next handler exists.
        """
        if self._next is None:
            return data
        return self._next.handle(data)


class CSVLoadHandler(Handler):
    """
    Loads a CSV file into a pandas DataFrame.
    """

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Parameters
        ----------
        data : Any
            Path to the CSV file.

        Returns
        -------
        pandas.DataFrame
            Loaded data with source path stored in ``DataFrame.attrs``.

        Raises
        ------
        FileNotFoundError
           If the CSV file does not exist.
        """
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig")

        # Store source path for later use
        df.attrs["__source_path__"] = str(path)
        logger.info("CSV loaded successfully from %s. Shape: %s", path, df.shape)

        # Debug information about loaded columns
        logger.debug("Columns: %s", df.columns.tolist())
        return self._call_next(df)


class InitialCleanupHandler(Handler):
    """
    Performs initial renaming and dropping of unnecessary columns.
    """

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Perform initial cleanup of raw data.

        This includes renaming inconsistent columns and removing
        unnecessary technical columns.

        Parameters
        ----------
        data : pandas.DataFrame
            Input raw DataFrame.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame with standardized column names.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        df = df.rename(columns={"Unnamed: 0": "ID"})
        df = df.rename(columns={"Последенее/нынешнее место работы": "Последнее место работы"})
        df = df.rename(columns={"Последеняя/нынешняя должность": "Последняя должность"})
        df = df.rename(columns={"Ищет работу на должность:": "Ищет работу на должность"})

        if "Обновление резюме" in df.columns:
            df = df.drop(columns=["Обновление резюме"])

        logger.info("Initial cleanup completed.")
        return self._call_next(df)


class CleanTextHandler(Handler):
    """
    Cleans textual data from unwanted unicode and control characters.
    """

    @staticmethod
    def _clean_value(value: Any) -> Any:
        """
        Clean a single text value from unwanted characters.

        Removes BOM markers, non-breaking spaces, and control characters
        while preserving common whitespace.

        Parameters
        ----------
        value : Any
            Input cell value.

        Returns
        -------
        Any
            Cleaned string or the original value if not applicable.
        """
        if pd.isna(value) or not isinstance(value, str):
            return value

        # Remove BOM, non-breaking spaces, and unwanted control characters
        value = value.replace("\ufeff", "").replace("\xa0", " ")

        cleaned = "".join(
            char for char in value
            if char >= " " or char in ("\t", "\n", "\r")
        )
        return cleaned.strip()

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Apply text cleaning to all string columns in the DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with cleaned text columns.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        text_columns = df.select_dtypes(include=[object]).columns

        for col in text_columns:
            df[col] = df[col].map(self._clean_value)

        logger.info("Text columns cleaned.")
        return self._call_next(df)


class GenderAgeHandler(Handler):
    """
    Parses 'Пол, возраст' column into separate 'Пол' and 'Возраст' columns.
    """

    @staticmethod
    def _parse_row(row: pd.Series) -> pd.Series:
        """
        Parse gender and age from a combined column.

        Parameters
        ----------
        row : pandas.Series
            DataFrame row containing the 'Пол, возраст' field.

        Returns
        -------
        pandas.Series
            Series with two elements: gender and age.
        """
        value = row.get("Пол, возраст")
        if pd.isna(value):
            return pd.Series([None, None])

        parts = [p.strip() for p in str(value).split(",")]
        if len(parts) < 2:
            return pd.Series([None, None])

        # Extract only gender and age from the field
        gender_raw = parts[0]

        age_match = re.search(r"\d+", parts[1])
        age = int(age_match.group()) if age_match else None

        return pd.Series([gender_raw, age])

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Split combined gender and age column into separate features.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with separate 'Пол' and 'Возраст' columns.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()

        if "Пол, возраст" in df.columns:
            # Create temporary columns
            df[["Пол", "Возраст"]] = df.apply(self._parse_row, axis=1)
            df = df.drop(columns=["Пол, возраст"])

        # Normalize Gender
        if "Пол" in df.columns:
            df["Пол"] = (
                df["Пол"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(config.GENDER_MAP)
            )

        logger.info("Gender and Age parsed.")
        return self._call_next(df)


class AgeSanityHandler(Handler):
    """
    Cleans Age:
    - clips to [AGE_MIN, AGE_MAX]
    - fills NaN with median
    """

    def handle(self, data: Any) -> pd.DataFrame:
        """
            Validate and clean the 'Возраст' column.

            Converts values to numeric, optionally clips to the configured range,
            and fills missing values with the median.

            Parameters
            ----------
            data : Any
                Input data. Must be a pandas.DataFrame.

            Returns
            -------
            pandas.DataFrame
                DataFrame with cleaned 'Возраст' values.
            """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        if "Возраст" in df.columns:
            df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce")

            if getattr(config, "AGE_CLIP_ENABLED", False):
                df["Возраст"] = df["Возраст"].clip(config.AGE_MIN, config.AGE_MAX)

            med = df["Возраст"].median()
            df["Возраст"] = df["Возраст"].fillna(med)

        logger.info("Age sanity applied.")
        return self._call_next(df)


class SalaryHandler(Handler):
    """
    Parses 'ЗП' (Salary) column, converting currencies to RUB.
    """

    @staticmethod
    def _parse_salary(cell: Any) -> Optional[float]:
        """
        Parse salary value and convert it to RUB.

        Parameters
        ----------
        cell : Any
            Raw salary value.

        Returns
        -------
        float or None
            Salary amount in RUB or None if parsing fails.
        """
        if pd.isna(cell):
            return None

        text = str(cell).lower()
        num_match = re.search(r"(\d[\d\s]*)", text)
        if not num_match:
            return None

        # Extract the first numeric amount; ignore ranges and textual qualifiers (e.g., "from/to").
        amount = float(num_match.group(1).replace(" ", ""))

        # Apply the first matching currency rate; default is RUB (rate = 1.0).
        rate = 1.0
        for currency, val in config.CURRENCY_RATES.items():
            if currency in text:
                rate = val
                break

        return amount * rate

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Parse and normalize salary values into a numeric column in RUB.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with normalized salary column.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()

        if "ЗП" in df.columns:
            df["ЗП_РУБ"] = df["ЗП"].apply(self._parse_salary)
            df["ЗП_РУБ"] = pd.to_numeric(df["ЗП_РУБ"], errors="coerce")

            # Round and cast to Int64 (nullable int)
            df["ЗП_РУБ"] = df["ЗП_РУБ"].round(0).astype("Int64")

            # Rename back to original
            df = df.drop(columns=["ЗП"])
            df = df.rename(columns={"ЗП_РУБ": "ЗП"})

        logger.info("Salary parsed and converted to RUB.")
        return self._call_next(df)


class TargetSanityHandler(Handler):
    """
    Cleans and stabilizes target salary:
    - drops NaN
    - filters by [MIN_SALARY_RUB, MAX_SALARY_RUB]
    - optional winsorization by quantiles
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        self._target = target

    def handle(self, data: Any) -> pd.DataFrame:
        """
            Clean and stabilize the target column.

            Converts the target to numeric, drops missing values, filters by the
            configured bounds, and optionally winsorizes extreme values using quantiles.

            Parameters
            ----------
            data : Any
                Input data. Must be a pandas.DataFrame.

            Returns
            -------
            pandas.DataFrame
                DataFrame with a cleaned target column.
            """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        if self._target not in df.columns:
            raise ValueError(f"Target column '{self._target}' not found")

        df[self._target] = pd.to_numeric(df[self._target], errors="coerce")
        df = df[df[self._target].notna()].reset_index(drop=True)

        # Remove obvious garbage like 1 ruble and extremely high values
        df = df[
            (df[self._target] >= config.MIN_SALARY_RUB)
            & (df[self._target] <= config.MAX_SALARY_RUB)
        ].reset_index(drop=True)

        # Winsorize by quantiles to reduce the impact of extreme outliers (requires enough rows).
        if getattr(config, "Y_WINSORIZE_ENABLED", False) and len(df) >= 20:
            lo = float(df[self._target].quantile(config.Y_WINSORIZE_LO_Q))
            hi = float(df[self._target].quantile(config.Y_WINSORIZE_HI_Q))
            df[self._target] = df[self._target].clip(lo, hi)

        logger.info("Target sanity applied. Rows remaining: %d", len(df))
        return self._call_next(df)


class CityHandler(Handler):
    """
    Parses 'Город' to extract city name, relocation readiness, and business trip readiness.
    """

    @staticmethod
    def _parse_city_value(value: Any) -> Tuple[Optional[str], bool, bool]:
        """
        Parse city-related information from a location string.

        Extracts city name, relocation readiness, and business trip readiness.

        Parameters
        ----------
        value : Any
            Raw city field value.

        Returns
        -------
        tuple
            (city, relocation_ready, business_trip_ready)
        """
        if pd.isna(value):
            return None, False, False

        value_str = str(value)
        value_lower = value_str.lower()

        # Translate city if in map
        for eng, rus in config.CITY_TRANSLATION_MAP.items():
            if eng in value_lower:
                if rus is None:
                    # Keep original value for non-localized entries (e.g. remote work)
                    pass
                else:
                    value_str = rus

        # Remove parentheses
        clean_value = re.sub(r"\(.*?\)", "", value_str)
        parts = [p.strip() for p in clean_value.split(",") if p.strip()]

        city_parts = []
        relocation = False
        trips = False

        for part in parts:
            part_lower = part.lower()

            if "хочу переехать" in part_lower:
                relocation = True
            elif "переезду" in part_lower and "не готов" not in part_lower:
                relocation = True
            elif "командировкам" in part_lower and "не готов" not in part_lower:
                trips = True
            else:
                city_parts.append(part)

        city = ", ".join(city_parts) if city_parts else None
        return city, relocation, trips

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Extract city and mobility features from the location column.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with parsed city and mobility features.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()

        if "Город" in df.columns:
            parsed_data = df["Город"].apply(self._parse_city_value).tolist()

            df["Город"] = [x[0] for x in parsed_data]
            df["Готовность к переезду"] = [x[1] for x in parsed_data]
            df["Готовность к командировкам"] = [x[2] for x in parsed_data]

            # Further cleaning of 'Город' (keep only first part before comma)
            df["Город"] = (
                df["Город"]
                .astype(str)
                .str.split(",")
                .str[0]
                .str.strip()
            )

        logger.info("City column parsed.")
        return self._call_next(df)


class OneHotTextHandler(Handler):
    """
    Generic handler for One-Hot encoding text columns based on keyword patterns.
    """

    def __init__(self, column: str, patterns: Dict[str, List[str]]):
        super().__init__()
        self.column = column
        self.patterns = patterns

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Apply keyword-based One-Hot encoding to a text column.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with generated One-Hot features.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()

        if self.column in df.columns:
            src = df[self.column].fillna("").str.lower()

            for key, keywords in self.patterns.items():
                col_name = f"{self.column}_{key}"
                df[col_name] = src.apply(lambda x: any(p in x for p in keywords))

            df = df.drop(columns=[self.column])

        logger.info("One-hot encoding completed for column: %s", self.column)
        return self._call_next(df)


class ExperienceHandler(Handler):
    """
    Parses experience strings into total months (integer).
    """

    @staticmethod
    def _parse_experience(value: Any) -> int:
        """
        Convert experience description into total number of months.

        Parameters
        ----------
        value : Any
            Raw experience description.

        Returns
        -------
        int
            Total experience in months.
        """
        if pd.isna(value):
            return 0

        text = str(value).lower()
        if "не указано" in text or "нет опыта" in text:
            return 0

        # Analyze header only
        header = text.split("\n")[0]
        years = 0
        months = 0

        # Regex for years
        y_match = re.search(r"(\d+)\s*(?:год|года|лет|year|years)", header)
        # Regex for months
        m_match = re.search(r"(\d+)\s*(?:месяц|месяца|месяцев|month|months)", header)

        if y_match:
            years = int(y_match.group(1))
        if m_match:
            months = int(m_match.group(1))

        total_months = years * 12 + months
        # Cap experience at 80 years to avoid outliers
        return min(total_months, 80 * 12)

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Parse work experience and convert it into numeric duration.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with experience expressed in months.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        col_name = "Опыт (двойное нажатие для полной версии)"

        if col_name in df.columns:
            df["Стаж (месяцев)"] = (
                df[col_name]
                .apply(self._parse_experience)
                .astype("int32")
            )
            df = df.drop(columns=[col_name])

        logger.info("Experience parsed into months.")
        return self._call_next(df)


class EducationHandler(Handler):
    """
    Parses 'Образование', extracts features, and classifies institutions.
    """

    @staticmethod
    def _parse_education_entry(value: Any) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Parse education entry into level, graduation year, and institution name.

        Parameters
        ----------
        value : Any
            Raw education string.

        Returns
        -------
        tuple
            (education_level, graduation_year, institution_name)
        """
        if pd.isna(value):
            return None, None, None

        # Format: "Level Year Institution"
        match = re.search(r"(.*?образование)\s+(\d{4})\s+(.*)", str(value))
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
        return None, None, None

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Parse education information and generate derived features.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with extracted education features.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        col_edu = "Образование и ВУЗ"

        if col_edu in df.columns:
            parsed = df[col_edu].apply(self._parse_education_entry).tolist()
            df["Уровень образования"] = [x[0] for x in parsed]
            df["Год окончания"] = [x[1] for x in parsed]
            df["Учебное заведение"] = [x[2] for x in parsed]
            df = df.drop(columns=[col_edu])

        # Clean 'Уровень образования'
        if "Уровень образования" in df.columns:
            df["Уровень образования"] = df["Уровень образования"].apply(
                lambda x: "Другое" if isinstance(x, str) and len(x) > 50 else x
            )

        # Generate Features from Institution Name
        if "Учебное заведение" in df.columns:
            src = df["Учебное заведение"].fillna("").str.lower()

            # 1. Category
            df["Категория_образования"] = None
            df["Категория_образования"] = df["Категория_образования"].astype("object")
            df.loc[src.str.contains(
                "|".join(config.EDUCATION_TECHNICAL_KEYWORDS)), "Категория_образования"] = "Технический"
            df.loc[src.str.contains(
                "|".join(config.EDUCATION_HUMANITARIAN_KEYWORDS)), "Категория_образования"] = "Гуманитарный"

            # 2. City of University
            df["Город_УЗ"] = None
            df["Город_УЗ"] = df["Город_УЗ"].astype("object")
            df.loc[src.str.contains("москва"), "Город_УЗ"] = "Москва"
            df.loc[src.str.contains("санкт-петербург|спб"), "Город_УЗ"] = "Санкт-Петербург"

            # 3. Specific flags
            df["IT_направление"] = src.str.contains(
                "информатик|программ|вычислительн|компьютер|сети", regex=True
            )
            df["Безопасность_направление"] = src.str.contains(
                "безопасност|защита|крипто", regex=True
            )

            # Cleanup temporary columns
            df = df.drop(columns=["Учебное заведение", "Год окончания"])

        logger.info("Education parsed and features generated.")
        return self._call_next(df)


class CardinalityReducerHandler(Handler):
    """
    Reduces cardinality of categorical columns by keeping only top N frequent values.
    """

    @staticmethod
    def _reduce_series(series: pd.Series, top_n: int) -> pd.Series:
        """
        Reduce categorical cardinality by keeping only top N frequent values.

        Parameters
        ----------
        series : pandas.Series
            Input categorical series.
        top_n : int
            Number of most frequent values to keep.

        Returns
        -------
        pandas.Series
            Series with infrequent values replaced by 'другое'.
        """
        top_values = series.value_counts().head(top_n).index
        return series.apply(lambda x: x if x in top_values else "другое")

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Reduce cardinality of selected categorical features.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with reduced-cardinality categorical features.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()

        for col in config.COLS_TO_REDUCE_CARDINALITY:
            if col in df.columns:
                # Normalize first
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = self._reduce_series(df[col], top_n=config.CARDINALITY_LIMIT)

        logger.info("Cardinality reduction completed.")
        return self._call_next(df)


class FillMissingHandler(Handler):
    """
    Handles missing values and filters rows without target variable.
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        self._target = target

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Handle missing values and filter invalid target rows.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with missing values handled.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        if self._target not in data.columns:
            raise ValueError(f"Target column '{self._target}' not found")

        df = data.copy()

        # Drop rows where target is NaN
        df = df[df[self._target].notna()].reset_index(drop=True)

        # Fill numeric cols with mean
        for col in df.select_dtypes(include=["number"]):
            df[col] = df[col].fillna(df[col].mean())

        # Fill categorical cols with placeholder
        for col in df.select_dtypes(include=[object]):
            df[col] = df[col].fillna("Не указано")

        df.attrs["__target__"] = self._target
        logger.info("Missing values filled. Rows remaining: %d", len(df))
        return self._call_next(df)


class DropColumnsHandler(Handler):
    """
    Drops columns that should not be used as model features (e.g., row ids).
    """

    def __init__(self, columns: list[str]) -> None:
        super().__init__()
        self._columns = columns

    def handle(self, data: Any) -> pd.DataFrame:
        """
           Drop configured columns if they exist in the DataFrame.

           Parameters
           ----------
           data : Any
               Input data. Must be a pandas.DataFrame.

           Returns
           -------
           pandas.DataFrame
               DataFrame with the specified columns removed (if present).
           """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        cols_to_drop = [c for c in self._columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info("Dropped columns: %s", cols_to_drop)
        return self._call_next(df)


class ExperienceMonthsSanityHandler(Handler):
    """
    Clips experience months to a reasonable upper bound.
    """

    def handle(self, data: Any) -> pd.DataFrame:
        """
            Validate and clip experience months to a reasonable range.

            Converts values to numeric and clips them to [0, EXP_MONTHS_MAX] if enabled.

            Parameters
            ----------
            data : Any
                Input data. Must be a pandas.DataFrame.

            Returns
            -------
            pandas.DataFrame
                DataFrame with cleaned experience months.
            """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = data.copy()
        col = "Стаж (месяцев)"
        if col in df.columns and getattr(config, "EXP_MONTHS_CLIP_ENABLED", False):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lower=0, upper=getattr(config, "EXP_MONTHS_MAX", 600))
        logger.info("Experience months sanity applied.")
        return self._call_next(df)


class EncodeHandler(Handler):
    """
    Encodes all categorical features using One-Hot Encoding (get_dummies).
    """

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Encode categorical features using One-Hot encoding.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            Fully numeric encoded DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        df = pd.get_dummies(data, drop_first=False)
        # Preserve attributes
        df.attrs.update(data.attrs)

        logger.info("Categorical encoding completed. Total columns: %d", df.shape[1])
        column_names = "\n".join(df.columns)
        logger.info("Column names:\n%s", column_names)

        return self._call_next(df)


class FeatureTargetSplitHandler(Handler):
    """
    Splits the DataFrame into X (features) and y (target) numpy arrays.
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        self._target = target

    def handle(self, data: Any) -> pd.DataFrame:
        """
        Split DataFrame into feature matrix X and target vector y.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            Original DataFrame with X and y stored in attributes.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        if self._target not in data.columns:
            raise ValueError(f"Target column '{self._target}' not found")

        # SAVE INTERMEDIATE CSV (X/y with column names)
        source = data.attrs.get("__source_path__")
        if source is not None:
            source_path = Path(source)

            data.to_csv(source_path.with_name("data_df.csv"), index=False, encoding="utf-8-sig")

            # 2) X with feature names
            X_df = data.drop(columns=[self._target])
            X_df.to_csv(source_path.with_name("X_data.csv"), index=False, encoding="utf-8-sig")

            # 3) y as separate csv
            data[[self._target]].to_csv(source_path.with_name("y_data.csv"), index=False, encoding="utf-8-sig")
        # END SAVE

        # Create Numpy arrays
        X = data.drop(columns=[self._target]).to_numpy(dtype=float)
        y = data[self._target].to_numpy(dtype=float)

        # Sanity checks
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Infinite values")

        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or Infinite values")

        data.attrs["__X__"] = X
        data.attrs["__y__"] = y

        logger.info("Data split into X: %s and y: %s", X.shape, y.shape)
        return self._call_next(data)


class SaveNpyHandler(Handler):
    """
    Saves the X and y arrays to .npy files.
    """

    def handle(self, data: Any) -> Tuple[str, str]:
        """
        Save feature matrix X and target vector y to .npy files.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing X and y stored in ``attrs``.

        Returns
        -------
        tuple of str
            Paths to saved X and y files.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas.DataFrame")

        X = data.attrs.get("__X__")
        y = data.attrs.get("__y__")
        source = data.attrs.get("__source_path__")

        if X is None or y is None or source is None:
            raise ValueError("Missing X, y, or source path for saving.")

        source_path = Path(source)
        x_path = source_path.with_name("x_data.npy")
        y_path = source_path.with_name("y_data.npy")

        np.save(x_path, X)
        np.save(y_path, y)

        logger.info("Successfully saved to %s and %s", x_path, y_path)
        return str(x_path), str(y_path)
