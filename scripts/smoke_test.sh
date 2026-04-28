#!/bin/bash
# scripts/smoke_test.sh
# Safely checks that the active runtime files exist and can be compiled.

echo "--- Running WSL CV Smoke Test ---"

# 1. Check active runtime files
FILES=("infer.py" "run_inference_sobal.py" "cut_img.py" "extract_track.py" "depth_adapter.py")
MISSING=0

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        MISSING=1
    else
        echo "✅ Found: $file"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "Smoke test failed: missing runtime files."
    exit 1
fi

# 2. Syntax check (compilation)
echo ""
echo "--- Compiling Active Python Files ---"
python -m py_compile infer.py run_inference_sobal.py cut_img.py extract_track.py depth_adapter.py

if [ $? -eq 0 ]; then
    echo "✅ All active files compiled successfully."
else
    echo "❌ Compilation failed."
    exit 1
fi

echo ""
echo "Smoke test passed. No inference was run."
