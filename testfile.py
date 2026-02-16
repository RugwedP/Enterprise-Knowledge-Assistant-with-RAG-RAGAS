from rag.loader import load_document

docs = load_document("data")

print("🔍 Searching for 'core hours' in documents...\n")

found = False
for i, doc in enumerate(docs):
    content = doc.page_content
    if "core hours" in content.lower() or "10:00 am" in content.lower() or "3:00 pm" in content.lower():
        found = True
        print(f"✓ Found in document {i}:")
        print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
        
        # Find and print the relevant section
        start = max(0, content.lower().find("core hours") - 100)
        end = min(len(content), content.lower().find("core hours") + 300)
        print(f"  Context: ...{content[start:end]}...")
        print()

if not found:
    print("❌ 'core hours' NOT found in any document!")
    print("\nℹ️  This means:")
    print("  1. The markdown files might not be in the data/ folder")
    print("  2. The content might use different wording")   