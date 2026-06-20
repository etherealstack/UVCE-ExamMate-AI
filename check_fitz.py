
try:
    import fitz
    print("✅ PyMuPDF (fitz) imported successfully")
except ImportError as e:
    print(f"❌ PyMuPDF Import failed: {e}")
