def get_base64_code(base64: str) -> str:
    result = ''
    for i in range(22, len(base64)):
        result += base64[i]
    return result