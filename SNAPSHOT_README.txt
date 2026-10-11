CODE-ONLY SNAPSHOT of the Farm Agent project (latest state).
Contains every source file, README.md, HANDOFF.md (read it first), requirements.txt, pods_manifest.json and dev/ tools.
It does NOT contain models/ (6 ONNX files), sample_images/, logs/ or dynamo_table.json - those stay in your full project folder.
To use: copy these files over your project folder (same layout), then run `python app_ui.py` in your venv.
A new assistant can read and edit all code from this snapshot; running the regression suite needs the full project (models + photos).
