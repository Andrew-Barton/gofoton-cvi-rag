import fitz
doc = fitz.open("data/sample_docs/North-America-Product-Catalog-2025_updated_compressed.pdf")
print(f"Pages: {len(doc)}")
print(doc[0].get_text())
