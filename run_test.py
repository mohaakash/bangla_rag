import subprocess
import sys

# Activate the virtual environment
activate_script = ".\\.venv\\Scripts\\activate.bat"

# The command to run the test script
question = "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
command = ["python", "test_code/test.py", "--text_path", "cleaned_text.txt", "--question", f'"{question}"']

# Run the command in a new shell with the virtual environment activated
process = subprocess.Popen(f"{activate_script} && {' '.join(command)}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Get the output
stdout, stderr = process.communicate()

# Write the output to a file
with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write(stdout.decode('utf-8'))
    f.write(stderr.decode('utf-8'))