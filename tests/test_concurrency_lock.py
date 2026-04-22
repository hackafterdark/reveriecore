import threading
import time
import pytest
import sqlite3
import os
from pathlib import Path
from reveriecore.database import DatabaseManager

def test_sqlite_concurrency_busy_timeout(tmp_path):
    """
    Simulates contention between two different connections to verify that 
    SQLite's busy_timeout (10s) allows the second operation to succeed after 
    the first one releases the lock.
    """
    db_path = str(tmp_path / "concurrency_test.db")
    
    # 1. Initialize DB through DatabaseManager
    db = DatabaseManager(db_path)
    with db.write_lock() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS stress_test (id INTEGER PRIMARY KEY, val TEXT)")
        cursor.execute("INSERT INTO stress_test (val) VALUES ('INIT')")
    
    results = []
    
    def blocker_thread():
        """
        Holds an IMMEDIATE transaction for 3 seconds using a raw connection.
        This bypasses DatabaseManager's threading.Lock.
        """
        try:
            # Separate connection to simulate another process/instance
            conn2 = sqlite3.connect(db_path)
            conn2.execute("PRAGMA journal_mode=WAL")
            # BEGIN IMMEDIATE starts a write transaction and prevents other writers
            conn2.execute("BEGIN IMMEDIATE")
            conn2.execute("UPDATE stress_test SET val = 'BLOCKED' WHERE id = 1")
            time.sleep(3) # Hold the lock
            conn2.commit()
            conn2.close()
            results.append("BLOCKER_DONE")
        except Exception as e:
            results.append(f"BLOCKER_FAIL: {e}")

    def waiter_thread():
        """
        Attempts a write using DatabaseManager. 
        It should wait for blocker_thread to release the lock.
        """
        time.sleep(0.5) # Ensure blocker_thread starts first
        start_time = time.time()
        try:
            with db.write_lock() as cursor:
                cursor.execute("UPDATE stress_test SET val = 'WAITER_WIN' WHERE id = 1")
            elapsed = time.time() - start_time
            results.append(f"WAITER_DONE_IN_{int(elapsed)}s")
        except Exception as e:
            results.append(f"WAITER_FAIL: {e}")

    t1 = threading.Thread(target=blocker_thread)
    t2 = threading.Thread(target=waiter_thread)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    # Assertions
    assert "BLOCKER_DONE" in results
    # Find the waiter result
    waiter_result = [r for r in results if r.startswith("WAITER_DONE")][0]
    assert "WAITER_DONE" in waiter_result
    
    # Verify the waiter actually succeeded after a delay
    # It should have waited for ~3 seconds
    assert "3s" in waiter_result or "2s" in waiter_result
    
    # Final value should be the waiter's value as it committed last
    cursor = db.get_cursor()
    cursor.execute("SELECT val FROM stress_test WHERE id = 1")
    assert cursor.fetchone()[0] == "WAITER_WIN"
    
    db.close()
