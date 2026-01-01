import re


def parse_response(response: str) -> str:
    if "<think>" in response and "</think>" in response:
        response = response.split("</think>")[1]

    if response is None:
        return ""

    if "```" not in response:
        return response

    code_pattern = r"```((.|\n)*?)```"

    if "```Python" in response:
        code_pattern = r"```Python((.|\n)*?)```"
    if "```Python3" in response:
        code_pattern = r"```Python3((.|\n)*?)```"
    if "```python" in response:
        code_pattern = r"```python((.|\n)*?)```"
    if "```python3" in response:
        code_pattern = r"```python3((.|\n)*?)```"

    code_blocks = re.findall(code_pattern, response, re.DOTALL)

    if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
        code_str = "\n".join(code_blocks[-1])
    elif type(code_blocks[-1]) == str:
        code_str = code_blocks[-1]
    else:
        code_str = response

    return code_str.strip()
