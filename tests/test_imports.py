# test_imports.py
import sys
import os
from pathlib import Path

print("🔍 Testing imports...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Try to import directly
try:
    from src.data_mapper import RAVDESSMapper
    print("✅ Successfully imported RAVDESSMapper")
except ImportError as e:
    print(f"❌ Failed to import RAVDESSMapper: {e}")

# Check if the file exists
data_mapper_path = Path("src/data_mapper.py")
if data_mapper_path.exists():
    print(f"✅ File exists: {data_mapper_path.absolute()}")
else:
    print(f"❌ File not found: {data_mapper_path.absolute()}")

# List all files in src directory
print("\n📂 Files in src directory:")
src_files = list(Path("src").glob("*.py"))
for file in src_files:
    print(f"   - {file.name}")