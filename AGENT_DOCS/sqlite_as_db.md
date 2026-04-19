We can use sqlite3 and the sqlite-vec extension to store embeddings in the database.

Example code:
```
import sqlite3
import sqlite_vec

# 1. Connect to your database file (or :memory:)
db = sqlite3.connect("reveries.db")

# 2. Load the extension into this specific connection
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# 3. You are now ready to use vector SQL commands!
# The 'vec0' virtual table is now available.
db.execute("CREATE VIRTUAL TABLE memories USING vec0(embedding float[384])")
```