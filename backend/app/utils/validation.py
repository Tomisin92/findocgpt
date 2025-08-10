# backend/app/utils/validation.py
"""
Input validation utilities for FinDocGPT API
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
import pandas as pd

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class Validator:
    """Comprehensive input validation utilities"""
    
    # Valid stock exchanges and their symbol patterns
    VALID_EXCHANGES = {
        'NYSE': r'^[A-Z]{1,4}$',
        'NASDAQ': r'^[A-Z]{1,5}$',
        'TSX': r'^[A-Z]{1,5}\.TO$',
        'LSE': r'^[A-Z]{1,4}\.L$',
        'ETF': r'^[A-Z]{2,5}$'
    }
    
    # Risk tolerance levels
    VALID_RISK_LEVELS = ['conservative', 'moderate', 'aggressive', 'very_conservative', 'very_aggressive']
    
    # Model types
    VALID_MODEL_TYPES = ['lstm', 'ensemble', 'arima', 'linear', 'random_forest']
    
    # Strategy types
    VALID_STRATEGIES = ['momentum', 'mean_reversion', 'trend_following', 'pairs_trading', 'buy_hold']
    
    # File types
    VALID_FILE_TYPES = ['.pdf', '.txt', '.docx', '.csv', '.xlsx']
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        symbol = symbol.upper().strip()
        
        # Check for basic format
        if not re.match(r'^[A-Z.]{1,10}$', symbol):
            raise ValidationError("Symbol must contain only uppercase letters and dots, max 10 characters")
        
        # Check against known patterns
        valid = False
        for exchange, pattern in Validator.VALID_EXCHANGES.items():
            if re.match(pattern, symbol):
                valid = True
                break
        
        if not valid:
            # Allow common index symbols
            index_symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^TNX']
            if symbol not in index_symbols:
                raise ValidationError(f"Symbol '{symbol}' does not match known exchange formats")
        
        return symbol
    
    @staticmethod
    def validate_symbols(symbols: Union[str, List[str]]) -> List[str]:
        """Validate multiple stock symbols"""
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        
        if not isinstance(symbols, list):
            raise ValidationError("Symbols must be a string or list of strings")
        
        if len(symbols) == 0:
            raise ValidationError("At least one symbol must be provided")
        
        if len(symbols) > 50:
            raise ValidationError("Maximum 50 symbols allowed")
        
        validated_symbols = []
        for symbol in symbols:
            validated_symbols.append(Validator.validate_symbol(symbol))
        
        return validated_symbols
    
    @staticmethod
    def validate_date(date_str: str, allow_future: bool = False) -> datetime:
        """Validate date string format"""
        if not date_str or not isinstance(date_str, str):
            raise ValidationError("Date must be a non-empty string")
        
        # Try multiple date formats
        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
        
        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            raise ValidationError("Date must be in format YYYY-MM-DD, MM/DD/YYYY, or DD/MM/YYYY")
        
        # Check if date is too old (before 1900)
        if parsed_date.year < 1900:
            raise ValidationError("Date cannot be before year 1900")
        
        # Check if date is in the future
        if not allow_future and parsed_date.date() > datetime.now().date():
            raise ValidationError("Date cannot be in the future")
        
        return parsed_date
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str, max_years: int = 10) -> tuple:
        """Validate date range"""
        start = Validator.validate_date(start_date)
        end = Validator.validate_date(end_date, allow_future=True)
        
        if start >= end:
            raise ValidationError("Start date must be before end date")
        
        # Check maximum range
        date_diff = end - start
        if date_diff.days > (max_years * 365):
            raise ValidationError(f"Date range cannot exceed {max_years} years")
        
        # Minimum range check
        if date_diff.days < 1:
            raise ValidationError("Date range must be at least 1 day")
        
        return start, end
    
    @staticmethod
    def validate_amount(amount: Any, min_value: float = 0.01, max_value: float = 1e12) -> float:
        """Validate monetary amount"""
        if amount is None:
            raise ValidationError("Amount cannot be None")
        
        try:
            # Handle string inputs
            if isinstance(amount, str):
                # Remove currency symbols and commas
                amount = re.sub(r'[$,]', '', amount.strip())
                amount = float(amount)
            else:
                amount = float(amount)
        except (ValueError, TypeError):
            raise ValidationError("Amount must be a valid number")
        
        if amount < min_value:
            raise ValidationError(f"Amount must be at least ${min_value:,.2f}")
        
        if amount > max_value:
            raise ValidationError(f"Amount cannot exceed ${max_value:,.2f}")
        
        # Check for reasonable precision (max 2 decimal places for currency)
        if round(amount, 2) != amount:
            amount = round(amount, 2)
        
        return amount
    
    @staticmethod
    def validate_percentage(percentage: Any, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """Validate percentage value"""
        try:
            if isinstance(percentage, str):
                # Remove % symbol
                percentage = percentage.strip().rstrip('%')
                percentage = float(percentage)
            else:
                percentage = float(percentage)
        except (ValueError, TypeError):
            raise ValidationError("Percentage must be a valid number")
        
        if percentage < min_val:
            raise ValidationError(f"Percentage must be at least {min_val}%")
        
        if percentage > max_val:
            raise ValidationError(f"Percentage cannot exceed {max_val}%")
        
        return percentage
    
    @staticmethod
    def validate_risk_tolerance(risk_tolerance: str) -> str:
        """Validate risk tolerance level"""
        if not risk_tolerance or not isinstance(risk_tolerance, str):
            raise ValidationError("Risk tolerance must be a non-empty string")
        
        risk_tolerance = risk_tolerance.lower().strip()
        
        if risk_tolerance not in Validator.VALID_RISK_LEVELS:
            raise ValidationError(f"Risk tolerance must be one of: {', '.join(Validator.VALID_RISK_LEVELS)}")
        
        return risk_tolerance
    
    @staticmethod
    def validate_model_type(model_type: str) -> str:
        """Validate ML model type"""
        if not model_type or not isinstance(model_type, str):
            raise ValidationError("Model type must be a non-empty string")
        
        model_type = model_type.lower().strip()
        
        if model_type not in Validator.VALID_MODEL_TYPES:
            raise ValidationError(f"Model type must be one of: {', '.join(Validator.VALID_MODEL_TYPES)}")
        
        return model_type
    
    @staticmethod
    def validate_strategy(strategy: str) -> str:
        """Validate trading strategy"""
        if not strategy or not isinstance(strategy, str):
            raise ValidationError("Strategy must be a non-empty string")
        
        strategy = strategy.lower().strip()
        
        if strategy not in Validator.VALID_STRATEGIES:
            raise ValidationError(f"Strategy must be one of: {', '.join(Validator.VALID_STRATEGIES)}")
        
        return strategy
    
    @staticmethod
    def validate_days(days: Any, min_days: int = 1, max_days: int = 365) -> int:
        """Validate number of prediction days"""
        try:
            days = int(days)
        except (ValueError, TypeError):
            raise ValidationError("Days must be a valid integer")
        
        if days < min_days:
            raise ValidationError(f"Days must be at least {min_days}")
        
        if days > max_days:
            raise ValidationError(f"Days cannot exceed {max_days}")
        
        return days
    
    @staticmethod
    def validate_confidence_level(confidence: Any) -> float:
        """Validate confidence level (0.0 to 1.0)"""
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            raise ValidationError("Confidence level must be a valid number")
        
        if confidence <= 0.0 or confidence >= 1.0:
            raise ValidationError("Confidence level must be between 0 and 1 (exclusive)")
        
        return confidence
    
    @staticmethod
    def validate_file_upload(filename: str, file_size: int, max_size_mb: int = 50) -> str:
        """Validate uploaded file"""
        if not filename or not isinstance(filename, str):
            raise ValidationError("Filename must be provided")
        
        filename = filename.strip()
        
        # Check file extension
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        
        if file_ext not in Validator.VALID_FILE_TYPES:
            raise ValidationError(f"File type must be one of: {', '.join(Validator.VALID_FILE_TYPES)}")
        
        # Check file size
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValidationError(f"File size cannot exceed {max_size_mb}MB")
        
        if file_size == 0:
            raise ValidationError("File cannot be empty")
        
        # Check for dangerous filenames
        dangerous_patterns = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
        for pattern in dangerous_patterns:
            if pattern in filename:
                raise ValidationError("Filename contains invalid characters")
        
        return filename
    
    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Validate portfolio allocation weights"""
        if not isinstance(weights, dict):
            raise ValidationError("Weights must be a dictionary")
        
        if len(weights) == 0:
            raise ValidationError("At least one weight must be provided")
        
        if len(weights) > 50:
            raise ValidationError("Maximum 50 positions allowed")
        
        validated_weights = {}
        total_weight = 0.0
        
        for symbol, weight in weights.items():
            # Validate symbol
            symbol = Validator.validate_symbol(symbol)
            
            # Validate weight
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                raise ValidationError(f"Weight for {symbol} must be a valid number")
            
            if weight < 0:
                raise ValidationError(f"Weight for {symbol} cannot be negative")
            
            if weight > 1:
                raise ValidationError(f"Weight for {symbol} cannot exceed 100%")
            
            validated_weights[symbol] = weight
            total_weight += weight
        
        # Check if weights sum to approximately 1
        if abs(total_weight - 1.0) > 0.01:
            raise ValidationError(f"Weights must sum to 100% (current sum: {total_weight:.1%})")
        
        return validated_weights
    
    @staticmethod
    def validate_json_data(data: Any, required_fields: List[str] = None) -> Dict:
        """Validate JSON data structure"""
        if not isinstance(data, dict):
            raise ValidationError("Data must be a valid JSON object")
        
        if required_fields:
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        return data
    
    @staticmethod
    def validate_pagination(page: Any = 1, limit: Any = 20, max_limit: int = 100) -> tuple:
        """Validate pagination parameters"""
        try:
            page = int(page)
            limit = int(limit)
        except (ValueError, TypeError):
            raise ValidationError("Page and limit must be valid integers")
        
        if page < 1:
            raise ValidationError("Page must be at least 1")
        
        if limit < 1:
            raise ValidationError("Limit must be at least 1")
        
        if limit > max_limit:
            raise ValidationError(f"Limit cannot exceed {max_limit}")
        
        return page, limit
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None, min_rows: int = 1) -> pd.DataFrame:
        """Validate pandas DataFrame"""
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame")
        
        if len(df) < min_rows:
            raise ValidationError(f"DataFrame must have at least {min_rows} rows")
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValidationError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for empty DataFrame
        if df.empty:
            raise ValidationError("DataFrame cannot be empty")
        
        return df
    
    @staticmethod
    def validate_api_key(api_key: str, service: str) -> str:
        """Validate API key format"""
        if not api_key or not isinstance(api_key, str):
            raise ValidationError(f"{service} API key must be provided")
        
        api_key = api_key.strip()
        
        # Basic format validation for common services
        if service.lower() == 'openai':
            if not api_key.startswith('sk-'):
                raise ValidationError("OpenAI API key must start with 'sk-'")
            if len(api_key) < 40:
                raise ValidationError("OpenAI API key appears to be too short")
        
        elif service.lower() == 'alpha_vantage':
            if len(api_key) < 8:
                raise ValidationError("Alpha Vantage API key appears to be too short")
        
        return api_key

class BulkValidator:
    """Utilities for validating multiple items at once"""
    
    @staticmethod
    def validate_bulk_symbols(symbols: List[str], max_symbols: int = 20) -> List[str]:
        """Validate multiple symbols with batch limits"""
        if len(symbols) > max_symbols:
            raise ValidationError(f"Maximum {max_symbols} symbols allowed per request")
        
        validated = []
        errors = []
        
        for i, symbol in enumerate(symbols):
            try:
                validated.append(Validator.validate_symbol(symbol))
            except ValidationError as e:
                errors.append(f"Symbol {i+1} ({symbol}): {str(e)}")
        
        if errors:
            raise ValidationError("Validation errors: " + "; ".join(errors))
        
        return validated
    
    @staticmethod
    def validate_bulk_amounts(amounts: List[Any]) -> List[float]:
        """Validate multiple amounts"""
        validated = []
        errors = []
        
        for i, amount in enumerate(amounts):
            try:
                validated.append(Validator.validate_amount(amount))
            except ValidationError as e:
                errors.append(f"Amount {i+1}: {str(e)}")
        
        if errors:
            raise ValidationError("Validation errors: " + "; ".join(errors))
        
        return validated

# Convenience functions for common validations
def validate_request_data(data: Dict, symbol_required: bool = True, amount_required: bool = False) -> Dict:
    """Validate common API request data"""
    validated = {}
    
    if symbol_required and 'symbol' in data:
        validated['symbol'] = Validator.validate_symbol(data['symbol'])
    elif symbol_required:
        raise ValidationError("Symbol is required")
    
    if amount_required and 'amount' in data:
        validated['amount'] = Validator.validate_amount(data['amount'])
    elif amount_required:
        raise ValidationError("Amount is required")
    
    # Optional fields
    if 'days' in data:
        validated['days'] = Validator.validate_days(data['days'])
    
    if 'model_type' in data:
        validated['model_type'] = Validator.validate_model_type(data['model_type'])
    
    if 'risk_tolerance' in data:
        validated['risk_tolerance'] = Validator.validate_risk_tolerance(data['risk_tolerance'])
    
    return validated

def sanitize_string(text: str, max_length: int = 1000) -> str:
    """Sanitize string input"""
    if not isinstance(text, str):
        raise ValidationError("Text must be a string")
    
    # Remove dangerous characters
    text = re.sub(r'[<>"\';()&+]', '', text)
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()