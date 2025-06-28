from functions.run_python import run_python_file

def run_tests():
    print(run_python_file("calculator", "main.py"), end="\n\n")

    print(run_python_file("calculator", "tests.py"), end="\n\n")

    print(run_python_file("calculator", "../main.py"), end="\n\n")

    print(run_python_file("calculator", "nonexistent.py"), end="\n\n")

if __name__ == "__main__":
    run_tests()
