import inspect
import tools.price_loader   # replace with your module name, e.g. tools.price_loader

def list_function_docstrings(module):
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        print(f"Function: {name}")
        print(f"Docstring: {inspect.getdoc(obj)}\n")

list_function_docstrings(tools.price_loader)