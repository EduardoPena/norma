"""Table context builder for LLM prompts"""
from typing import Dict, Any, Literal
import json
import pandas as pd


SampleStrategy = Literal["head", "random", "full", "informative"]


class TableContextBuilder:
    """Builds formatted table context for LLM prompts."""
    
    def __init__(self,
                 sample_strategy: SampleStrategy = "head",
                 num_rows: int = 3,
                 column_names_only: bool = False,
                 random_seed: int = 42):
        """
        Args:
            sample_strategy: How to sample table rows ("head", "random", "full",
                "informative" = rows that together fill the most columns)
            num_rows: Number of rows to sample
            column_names_only: If True, only column names (no data rows)
            random_seed: Seed for reproducible random sampling
        """
        self.sample_strategy = sample_strategy
        self.num_rows = num_rows
        self.column_names_only = column_names_only
        self.random_seed = random_seed
    
    def build(self, table: pd.DataFrame) -> Dict[str, Any]:
        """
        Build context from table.
        
        Args:
            table: DataFrame to build context from
            
        Returns:
            Context dictionary with "table_context" key
        """
        return {
            "table_context": self._format_table(table)
        }
    
    def _format_table(self, table: pd.DataFrame) -> str:
        """Format table as string for LLM context."""
        if self.column_names_only:
            return "Columns: " + ", ".join(table.columns)
        
        sample = self._sample_table(table)
        
        table_data = {
            "columns": list(sample.columns),
            "rows": [
                {column: self._serialize_value(row[column]) for column in sample.columns}
                for _, row in sample.iterrows()
            ]
        }
        
        return "Table data (JSON format):\n" + json.dumps(table_data, ensure_ascii=False)
    
    def _sample_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Sample DataFrame according to strategy."""
        if self.sample_strategy == "head":
            return table.head(self.num_rows)
        
        elif self.sample_strategy == "random":
            actual_rows = min(self.num_rows, len(table))
            return table.sample(n=actual_rows, random_state=self.random_seed)
        
        elif self.sample_strategy == "full":
            return table

        elif self.sample_strategy == "informative":
            return self._informative_rows(table)
        
        else:
            raise ValueError(f"Unknown sample_strategy: {self.sample_strategy}")
    
    def _informative_rows(self, table: pd.DataFrame, pool_size: int = 5000) -> pd.DataFrame:
        """Greedy pick of rows that cover the most still-empty columns.

        With wide, sparse tables the first rows can leave many columns empty,
        hiding the columns the rules talk about. Rows come from the first
        ``pool_size`` rows; ties keep the earlier row, so the choice is
        deterministic.
        """
        pool = table.head(pool_size)
        filled = pool.notna().to_numpy()
        chosen: list = []
        covered = None
        for _ in range(min(self.num_rows, len(pool))):
            gain = filled.sum(axis=1) if covered is None else (filled & ~covered).sum(axis=1)
            for index in chosen:
                gain[index] = -1
            best = int(gain.argmax())
            chosen.append(best)
            covered = filled[best] if covered is None else covered | filled[best]
        return pool.iloc[sorted(chosen)]

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize pandas value to JSON-compatible format."""
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):  # datetime
            return value.isoformat()
        return value