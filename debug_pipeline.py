# debug_pipeline.py
import sys
import traceback

print("="*60)
print("🔍 DEBUGGING TRAINING PIPELINE")
print("="*60)

try:
    print("1. Importing TrainingPipeline...")
    from src.training_pipeline import TrainingPipeline
    print("✅ Import successful")
    
    print("\n2. Creating pipeline instance...")
    pipeline = TrainingPipeline()
    print("✅ Pipeline created")
    
    print("\n3. Running test with 2 files...")
    summary = pipeline.run_pipeline(max_files=2)
    print("✅ Pipeline run complete")
    
    if summary:
        print(f"\n📊 Summary: {summary.keys()}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nDetailed traceback:")
    traceback.print_exc()
    
print("\n" + "="*60)