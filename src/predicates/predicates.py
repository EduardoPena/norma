"""Predicate representation for denial constraints"""
from dataclasses import dataclass
from typing import Optional, Dict, Set


@dataclass(frozen=True)
class Predicate:
    """Represents a single predicate like 'p1.zip = p2.zip' or 'p1.age > 18'"""
    left: str      # e.g., 'p1.zip'
    op: str        # e.g., '=', '!=', '<', etc.
    right: str     # e.g., 'p2.zip' or '18'
    
    @classmethod
    def from_string(cls, pred_str: str) -> 'Predicate':
        """Parse 'p1.zip = p2.zip' into Predicate"""
        # Order matters! Check multi-character operators first
        # IS NOT NULL and IS NULL are special - they don't have a right operand
        for op in ['IS NOT NULL', 'IS NULL']:
            if pred_str.endswith(f' {op}'):
                left = pred_str[:-len(f' {op}')].strip()
                return cls(left, op, 'NULL')
        
        for op in ['NOT LIKE', 'NOT IN', 'LIKE', 'IN', '<=', '>=', '!=', '=', '<', '>']:
            if f' {op} ' in pred_str:
                left, right = pred_str.split(f' {op} ', 1)
                left = left.strip()
                right = right.strip()
                
                # Normalize quotes: remove surrounding quotes from literals
                # This ensures 'Assis', "Assis", and Assis are treated the same
                if (right.startswith("'") and right.endswith("'")) or \
                   (right.startswith('"') and right.endswith('"')):
                    right = right[1:-1]
                
                return cls(left, op, right)
        raise ValueError(f"Invalid predicate: {pred_str}")
    
    def to_string(self) -> str:
        """Convert back to string"""
        # Handle IS NULL and IS NOT NULL specially
        if self.op in ['IS NULL', 'IS NOT NULL']:
            return f"{self.left} {self.op}"
        
        # Add quotes to string literals (anything that's not a column reference or number)
        right = self.right
        if not right.replace('.', '').replace('-', '').replace('_', '').isalnum():
            # Not a simple column/variable name or number, add quotes
            if not (right.startswith("'") or right.startswith('"')):
                right = f"'{right}'"
        elif '.' not in right and not right.replace('.', '').replace('-', '').isdigit():
            # String literal without quotes - add them
            try:
                float(right)
            except ValueError:
                # Not a number, likely a string literal
                right = f"'{right}'"
        
        return f"{self.left} {self.op} {right}"
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __eq__(self, other) -> bool:
        """Two predicates are equal if all fields match"""
        if not isinstance(other, Predicate):
            return False
        return self.left == other.left and self.op == other.op and self.right == other.right
    
    def __hash__(self) -> int:
        """Make hashable for set operations"""
        return hash((self.left, self.op, self.right))


@dataclass
class PredicateSpace:
    """Encapsulates all predicate space information from predicates_spec"""
    quantifiers: Dict[str, str]  # {'p1': 'person', 'p2': 'person'}
    allowed_predicates: Set[str]  # Set of allowed predicate strings
    schema: Dict[str, Dict[str, Dict]]  # table -> column -> {type, allowed_ops}
    all_columns: Set[str]  # Flat set of all column names
    
    @classmethod
    def from_spec(cls, predicates_spec: Dict) -> 'PredicateSpace':
        """Create PredicateSpace from predicates specification"""
        quantifiers = predicates_spec['quantifiers']
        allowed_predicates = set(predicates_spec['predicates'])
        
        # Extract schema
        schema = {}
        for table in predicates_spec['tables']:
            schema[table['name']] = {
                col['name']: {
                    'type': col['type'],
                    'allowed_ops': col['allowed_ops']
                }
                for col in table['columns']
            }
        
        # Flat column list
        all_columns = set()
        for table_cols in schema.values():
            all_columns.update(table_cols.keys())
        
        return cls(quantifiers, allowed_predicates, schema, all_columns)
    
    def is_valid_predicate(self, pred_str: str) -> bool:
        """Check if predicate is valid"""
        if pred_str in self.allowed_predicates:
            return True
        
        # Check constant instantiation
        pred = Predicate.from_string(pred_str)
        
        # Extract the quantifier and column from the left side (e.g., "r1.grade" -> "r1", "grade")
        if '.' in pred.left:
            pred_quantifier, pred_column = pred.left.split('.', 1)
        else:
            return False
        
        for allowed in self.allowed_predicates:
            allowed_pred = Predicate.from_string(allowed)
            
            # Extract quantifier and column from allowed predicate
            if '.' in allowed_pred.left:
                allowed_quantifier, allowed_column = allowed_pred.left.split('.', 1)
            else:
                continue
            
            # Check if columns and operators match (quantifiers can be different)
            if pred_column == allowed_column and pred.op == allowed_pred.op:
                # Check if the allowed predicate has a wildcard placeholder
                if allowed_pred.right.startswith('$'):
                    return True
        
        return False
    

