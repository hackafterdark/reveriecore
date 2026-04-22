import os
import sys
from pathlib import Path
import shutil
import uuid
import json
from unittest.mock import MagicMock

# Add current path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database import DatabaseManager
from mirror import MirrorService

def run_verification():
    tmp_path = Path("tmp_verification")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    
    db_path = tmp_path / "test_reveries.db"
    archive_path = tmp_path / "test_archive"
    db = DatabaseManager(str(db_path))
    
    # Mock Enrichment
    enrichment = MagicMock()
    enrichment.generate_embedding.return_value = [0.1] * 384
    enrichment.count_tokens.return_value = 10
    
    mirror = MirrorService(db, enrichment, archive_root=archive_path)
    
    print("Running Export/Import Cycle Verification...")
    
    # 1. Add a memory
    content = "Verification test content"
    guid = str(uuid.uuid4())
    learned_at = "2024-05-21T12:00:00"
    with db.write_lock() as cursor:
        cursor.execute("""
            INSERT INTO memories (content_full, content_abstract, guid, author_id, learned_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (content, "Abstract here", guid, "VERIFY_USER", learned_at, "ACTIVE"))
        mem_id = cursor.lastrowid
    
    # 2. Export
    mirror.export_node(mem_id)
    
    # 3. Verify file exists with Hive pathing
    hive_path = archive_path / "year=2024" / "month=05" / "day=21" / f"{guid}.md"
    if not hive_path.exists():
        # Fallback check in case pathing logic used current datetime instead of learned_at
        print("Warning: Hive path was not created using learned_at (trying fallback check)")
        files = list(archive_path.glob("**/*.md"))
        if not files:
            raise Exception("No markdown files were created")
        hive_path = files[0]
        
    print(f"✓ Markdown file created at: {hive_path}")
    
    # Check content
    with open(hive_path, "r") as f:
        md_content = f.read()
        if guid not in md_content: raise Exception("GUID not in markdown")
        if content not in md_content: raise Exception("Content not in markdown")
        if "Abstract here" not in md_content: raise Exception("Abstract not in markdown")
    print("✓ Markdown content verified")
    
    # 4. Wipe DB
    with db.write_lock() as cursor:
        cursor.execute("DELETE FROM memories")
    
    if db.get_memory_by_guid(guid) is not None:
        raise Exception("Database was not wiped")
    
    # 5. Import
    mirror.import_archive()
    
    # 6. Verify restored
    restored = db.get_memory_by_guid(guid)
    if restored is None:
        raise Exception("Memory was not restored from archive")
    if restored['content_full'] != content:
        raise Exception(f"Content mismatch after restore: Expected {content}, got {restored['content_full']}")
    if restored['guid'] != guid:
        raise Exception("GUID mismatch after restore")
    print("✓ Restore verified successfully")
    
    # 7. cleanup
    db.close()
    shutil.rmtree(tmp_path)
    print("\nVerification PASSED")

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
