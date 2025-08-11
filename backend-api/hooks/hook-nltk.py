# Custom hook to handle NLTK sqlite3 dependency issue
import sys
import types

# Create a mock sqlite3 module
mock_sqlite3 = types.ModuleType('sqlite3')
mock_sqlite3.__file__ = '<mock>'

# Add basic sqlite3 functionality that NLTK might need
class MockConnection:
    def __init__(self, *args, **kwargs):
        pass
    
    def execute(self, *args, **kwargs):
        return MockCursor()
    
    def commit(self):
        pass
    
    def close(self):
        pass

class MockCursor:
    def __init__(self):
        pass
    
    def fetchall(self):
        return []
    
    def fetchone(self):
        return None

mock_sqlite3.connect = MockConnection
mock_sqlite3.Connection = MockConnection
mock_sqlite3.Cursor = MockCursor

# Inject the mock module
sys.modules['sqlite3'] = mock_sqlite3 