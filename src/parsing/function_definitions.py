def get_function_definitions(path: str) -> list[dict]:
    function_lists = []
    with open(path, 'r') as f:
        function_definitions = json.load(f)
    for function_definition in function_definitions:
        function_lists.append(function_definition)
    return function_lists