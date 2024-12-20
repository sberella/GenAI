import os

def load_env(file_path=".env"):
    with open(file_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Example usage
# load_env()
# print(os.getenv("MY_VARIABLE"))
