import string
import secrets

total_characters = string.ascii_letters + string.digits
session_key_length = 32 + secrets.randbelow(len(total_characters) - 32)

session_key = "".join(secrets.choice(total_characters) for _ in range(session_key_length))

print(session_key_length)
print(session_key)
