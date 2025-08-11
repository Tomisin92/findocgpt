# Complete Document Processor for FinDocGPT
# Handles TXT, PDF, and CSV files with comprehensive financial analysis

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from typing import Dict, List, Any
from datetime import datetime

# PDF processing imports (install with: pip install PyPDF2 pdfplumber)
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PDF processing libraries not installed. Run: pip install PyPDF2 pdfplumber")

class DocumentProcessor:
    """Complete document processor for TXT, PDF, and CSV files in FinDocGPT"""
    
    @staticmethod
    def extract_document_text(uploaded_file) -> str:
        """Main entry point - extracts text from TXT, PDF, or CSV files"""
        try:
            file_type = uploaded_file.type
            file_name = uploaded_file.name.lower()
            
            # Determine file type
            if file_type == "text/plain" or file_name.endswith('.txt'):
                return DocumentProcessor._process_txt_file(uploaded_file)
            
            elif file_type == "application/pdf" or file_name.endswith('.pdf'):
                if PDF_AVAILABLE:
                    return DocumentProcessor._process_pdf_file(uploaded_file)
                else:
                    return "❌ PDF processing not available. Install required libraries: pip install PyPDF2 pdfplumber"
            
            elif file_type in ["text/csv", "application/vnd.ms-excel"] or file_name.endswith('.csv'):
                return DocumentProcessor._process_csv_file(uploaded_file)
            
            else:
                return f"❌ Unsupported file type: {file_type}. Please upload .txt, .pdf, or .csv files only."
                
        except Exception as e:
            return f"❌ Error processing file: {str(e)}"
    
    @staticmethod
    def _process_txt_file(uploaded_file) -> str:
        """Process TXT files with encoding detection and analysis"""
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            used_encoding = 'utf-8'
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode(encoding)
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return "❌ Could not decode text file. Please check the file encoding."
            
            if not content.strip():
                return "❌ The text file appears to be empty."
            
            # Clean content
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Extract financial metrics
            financial_metrics = DocumentProcessor._extract_financial_metrics(content)
            
            # Generate analysis
            lines = content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            word_count = len(content.split())
            
            analysis = f"""📄 TXT Document Analysis:
📊 File Statistics:
- Characters: {len(content):,}
- Words: {word_count:,}
- Lines: {len(lines):,} (non-empty: {len(non_empty_lines):,})
- Encoding: {used_encoding}
- Reading time: ~{word_count // 200} minutes

💰 Financial Data Detected:
{DocumentProcessor._format_financial_metrics(financial_metrics)}

📝 Document Content:
{content}"""
            
            return analysis
            
        except Exception as e:
            return f"❌ Error processing TXT file: {str(e)}"
    
    @staticmethod
    def _process_pdf_file(uploaded_file) -> str:
        """Process PDF files with multiple extraction methods"""
        try:
            # Method 1: pdfplumber (better for financial documents)
            try:
                pdf_bytes = uploaded_file.read()
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    text_pages = []
                    tables_found = 0
                    
                    # Limit to first 50 pages for performance
                    for i, page in enumerate(pdf.pages[:50]):
                        page_text = page.extract_text()
                        if page_text:
                            text_pages.append(f"--- Page {i+1} ---\n{page_text}")
                        
                        # Check for tables
                        tables = page.extract_tables()
                        if tables:
                            tables_found += len(tables)
                    
                    if text_pages:
                        full_text = "\n\n".join(text_pages)
                        financial_metrics = DocumentProcessor._extract_financial_metrics(full_text)
                        
                        analysis = f"""📄 PDF Document Analysis:
📊 File Statistics:
- Pages processed: {len(text_pages)}/{len(pdf.pages)}
- Characters extracted: {len(full_text):,}
- Tables detected: {tables_found}
- Word count: ~{len(full_text.split()):,}

💰 Financial Data Detected:
{DocumentProcessor._format_financial_metrics(financial_metrics)}

📝 Document Content:
{full_text}"""
                        
                        return analysis
                        
            except Exception as e:
                st.warning(f"pdfplumber failed: {str(e)}, trying PyPDF2...")
            
            # Method 2: PyPDF2 fallback
            uploaded_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_pages = []
            
            for page_num, page in enumerate(pdf_reader.pages[:50]):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(f"--- Page {page_num+1} ---\n{page_text}")
            
            if text_pages:
                full_text = "\n\n".join(text_pages)
                financial_metrics = DocumentProcessor._extract_financial_metrics(full_text)
                
                analysis = f"""📄 PDF Document Analysis (PyPDF2):
📊 File Statistics:
- Pages processed: {len(text_pages)}/{len(pdf_reader.pages)}
- Characters extracted: {len(full_text):,}
- Word count: ~{len(full_text.split()):,}

💰 Financial Data Detected:
{DocumentProcessor._format_financial_metrics(financial_metrics)}

📝 Document Content:
{full_text}"""
                
                return analysis
            else:
                return "❌ Could not extract text from PDF. File may be image-based or encrypted."
                
        except Exception as e:
            return f"❌ Error processing PDF file: {str(e)}"
    
    @staticmethod
    def _process_csv_file(uploaded_file) -> str:
        """Process CSV files with comprehensive financial analysis"""
        try:
            # Try different separators and encodings
            separators = [',', ';', '\t', '|']
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            df = None
            used_sep = ','
            used_encoding = 'utf-8'
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
                        if len(df.columns) > 1 and len(df) > 0:
                            used_sep = sep
                            used_encoding = encoding
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            
            if df is None or len(df.columns) <= 1:
                return "❌ Could not parse CSV file. Please check format and encoding."
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Analyze the dataset
            rows, cols = df.shape
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Identify financial columns
            financial_keywords = ['revenue', 'sales', 'income', 'profit', 'cost', 'expense', 
                                'assets', 'liability', 'cash', 'price', 'amount', 'value', 
                                'total', 'net', 'gross', 'operating', 'ebitda', 'margin']
            
            financial_cols = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in financial_keywords):
                    financial_cols.append(col)
            
            # Identify date columns
            date_cols = []
            for col in text_cols[:10]:  # Check first 10 text columns
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    date_cols.append(col)
                except:
                    continue
            
            # Calculate statistics for numeric columns
            numeric_stats = {}
            for col in numeric_cols[:10]:  # Limit to first 10
                try:
                    stats = df[col].describe()
                    numeric_stats[col] = {
                        'mean': stats['mean'],
                        'median': df[col].median(),
                        'min': stats['min'],
                        'max': stats['max'],
                        'null_count': df[col].isnull().sum()
                    }
                except:
                    continue
            
            # Data quality metrics
            total_cells = rows * cols
            missing_cells = df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            
            # Generate comprehensive analysis
            analysis = f"""📊 CSV Dataset Analysis:

📈 Dataset Overview:
- Rows: {rows:,}
- Columns: {cols}
- File encoding: {used_encoding}
- Separator: '{used_sep}'
- Memory usage: ~{df.memory_usage(deep=True).sum() / 1024:.1f} KB

📋 Column Types:
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:8])}{'...' if len(numeric_cols) > 8 else ''}
- Text columns ({len(text_cols)}): {', '.join(text_cols[:8])}{'...' if len(text_cols) > 8 else ''}
- Date columns ({len(date_cols)}): {', '.join(date_cols) if date_cols else 'None detected'}

💰 Financial Analysis:
- Financial columns identified ({len(financial_cols)}): {', '.join(financial_cols) if financial_cols else 'None detected'}"""

            # Add financial column statistics
            if financial_cols and numeric_stats:
                analysis += "\n\n📊 Key Financial Metrics:"
                for col in financial_cols[:5]:  # First 5 financial columns
                    if col in numeric_stats:
                        stats = numeric_stats[col]
                        analysis += f"""
• {col}:
  - Average: {stats['mean']:,.2f}
  - Median: {stats['median']:,.2f}
  - Range: {stats['min']:,.2f} to {stats['max']:,.2f}
  - Missing: {stats['null_count']} values"""
            
            # Data quality section
            analysis += f"""

📊 Data Quality:
- Completeness: {completeness:.1f}% ({missing_cells:,} missing out of {total_cells:,})
- Duplicate rows: {df.duplicated().sum():,}
- Unique values per column: {dict(df.nunique().head(5))}"""

            # Analysis potential
            analysis += "\n\n🎯 Analysis Potential:"
            if financial_cols:
                analysis += f"\n✅ High financial analysis potential - {len(financial_cols)} financial columns"
            if date_cols:
                analysis += f"\n✅ Time series analysis possible - Date columns available"
            if len(numeric_cols) >= 3:
                analysis += f"\n✅ Statistical analysis ready - {len(numeric_cols)} numeric columns"
            
            # Sample data
            analysis += f"""

📋 Sample Data (First 5 rows):
{df.head().to_string()}

📝 Full Dataset for Analysis:
{df.to_string()}"""

            return analysis
            
        except Exception as e:
            return f"❌ Error processing CSV file: {str(e)}"
    
    @staticmethod
    def _extract_financial_metrics(text: str) -> Dict[str, float]:
        """Extract financial metrics from text using regex patterns"""
        metrics = {}
        
        # Comprehensive financial patterns
        patterns = {
            'revenue': [
                r'(?:total\s+)?(?:net\s+)?revenue[s]?[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?',
                r'(?:net\s+)?sales[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?',
                r'total\s+revenue[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?'
            ],
            'net_income': [
                r'net\s+(?:income|earnings)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?',
                r'(?:net\s+)?profit[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?'
            ],
            'total_assets': [
                r'total\s+assets[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?'
            ],
            'operating_income': [
                r'operating\s+(?:income|earnings)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?'
            ],
            'cash': [
                r'cash\s+and\s+(?:cash\s+)?equivalents[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?',
                r'(?:total\s+)?cash[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?'
            ],
            'research_development': [
                r'research\s+and\s+development[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?',
                r'r&d[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|thousand|m|b|k)?'
            ]
        }
        
        text_lower = text.lower()
        
        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text_lower)
                if matches:
                    try:
                        value = float(matches[0].replace(',', ''))
                        # Convert to actual value if units specified
                        if 'billion' in text_lower or ' b' in text_lower:
                            value *= 1_000_000_000
                        elif 'million' in text_lower or ' m' in text_lower:
                            value *= 1_000_000
                        elif 'thousand' in text_lower or ' k' in text_lower:
                            value *= 1_000
                        
                        metrics[metric] = value
                        break
                    except ValueError:
                        continue
        
        return metrics
    
    @staticmethod
    def _format_financial_metrics(metrics: Dict[str, float]) -> str:
        """Format financial metrics for display"""
        if not metrics:
            return "No quantifiable financial metrics detected"
        
        formatted = []
        for metric, value in metrics.items():
            formatted_name = metric.replace('_', ' ').title()
            if value >= 1_000_000_000:
                formatted.append(f"• {formatted_name}: ${value/1_000_000_000:.1f}B")
            elif value >= 1_000_000:
                formatted.append(f"• {formatted_name}: ${value/1_000_000:.1f}M")
            elif value >= 1_000:
                formatted.append(f"• {formatted_name}: ${value/1_000:.1f}K")
            else:
                formatted.append(f"• {formatted_name}: ${value:,.0f}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def process_financial_document(document_text: str, query: str) -> str:
        """Process financial documents with AI analysis optimized for different file types"""
        if not document_text.strip():
            return "❌ No document content provided for analysis."
        
        # Check for processing errors
        if document_text.startswith("❌"):
            return document_text
        
        # Detect document type
        doc_type = "general"
        if "CSV Dataset Analysis:" in document_text:
            doc_type = "csv"
        elif "TXT Document Analysis:" in document_text:
            doc_type = "txt"  
        elif "PDF Document Analysis:" in document_text:
            doc_type = "pdf"
        
        # Create enhanced context for AI
        context = f"""
        FINDOCGPT DOCUMENT ANALYSIS
        Document Type: {doc_type.upper()}
        User Query: {query}
        
        Processed Document Content:
        {document_text[:2000]}...
        """
        
        # Generate type-specific prompts for better AI responses
        if doc_type == "csv":
            prompt = f"""
            As FinDocGPT analyzing a CSV financial dataset, answer: "{query}"
            
            Focus on:
            1. **Data Overview**: What financial/business data is contained?
            2. **Key Metrics**: Extract specific numbers and calculations
            3. **Trends & Patterns**: Identify significant patterns in the data
            4. **Financial Insights**: Calculate ratios, growth rates, or comparative analysis
            5. **Business Intelligence**: What actionable insights can be derived?
            
            CSV Analysis Details: {document_text}
            """
        
        elif doc_type == "txt":
            prompt = f"""
            As FinDocGPT analyzing a text financial document, answer: "{query}"
            
            Provide analysis including:
            1. **Direct Answer**: Address the specific query with exact figures
            2. **Supporting Evidence**: Quote relevant sections from the document  
            3. **Financial Metrics**: Extract and analyze quantitative data
            4. **Risk Assessment**: Identify risks, opportunities, or concerns mentioned
            5. **Business Context**: Explain implications for stakeholders
            
            Document Analysis: {document_text}
            """
            
        elif doc_type == "pdf":
            prompt = f"""
            As FinDocGPT analyzing a PDF financial document (likely 10-K, earnings report, or financial statement), answer: "{query}"
            
            Comprehensive analysis should include:
            1. **Specific Answer**: Address the query with precise data from the document
            2. **Financial Data**: Extract relevant metrics, ratios, and performance indicators
            3. **Document Context**: Identify document type and reporting period
            4. **Comparative Analysis**: Compare current vs previous periods if data available
            5. **Investment Implications**: What does this mean for investors/analysts?
            
            PDF Content Analysis: {document_text}
            """
        
        else:
            prompt = f"""
            As FinDocGPT, provide comprehensive financial analysis for: "{query}"
            Document: {document_text}
            """
        
        # Use existing OpenAI handler from main application
        try:
            response = OpenAIHandler.generate_response(prompt, context, "document_qa")
            
            # Add document type indicator to response
            type_headers = {
                "csv": "## 📊 CSV Financial Data Analysis\n\n",
                "txt": "## 📄 Text Document Financial Analysis\n\n", 
                "pdf": "## 📋 PDF Document Financial Analysis\n\n"
            }
            
            header = type_headers.get(doc_type, "## 🤖 FinDocGPT Analysis\n\n")
            return header + response
            
        except Exception as e:
            return f"❌ Error generating AI analysis: {str(e)}"

# Installation requirements for requirements.txt:
"""
# Core requirements
streamlit
pandas
numpy

# PDF processing (required for PDF support)
PyPDF2==3.0.1
pdfplumber==0.9.0

# Financial analysis
yfinance
plotly

# AI processing  
openai
"""

# Quick integration guide:
"""
INTEGRATION STEPS:

1. Install dependencies:
   pip install PyPDF2 pdfplumber

2. Replace the DocumentProcessor class in your FinDocGPT code with this complete version

3. Update your file upload section to show supported formats:
   st.file_uploader("Upload Financial Document", type=['txt', 'pdf', 'csv'])

4. Test with your files:
   - TXT: Financial reports, earnings transcripts
   - PDF: 10-K filings, annual reports  
   - CSV: Financial datasets, performance data

FEATURES:
✅ TXT: Multi-encoding support, financial metric extraction, word count analysis
✅ PDF: Multi-method extraction, table detection, page-by-page processing  
✅ CSV: Auto-format detection, financial column identification, statistical analysis
✅ All: Comprehensive error handling, optimized AI prompts, financial focus
"""