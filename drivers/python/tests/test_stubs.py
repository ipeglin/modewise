"""The type stubs are the Python documentation's single source: every exported name needs a
stub with a docstring, and each function's parameters must match the compiled one."""

import ast
import inspect
from pathlib import Path

import modewise

STUBS = Path(modewise.__file__).with_name("_core.pyi")


def stub_definitions():
    tree = ast.parse(STUBS.read_text())
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def test_every_exported_name_has_a_documented_stub():
    stubs = stub_definitions()
    # Names written in Python carry their own docstrings; __version__ is a constant.
    in_python = {n for n in modewise.__all__ if n not in stubs and inspect.getdoc(getattr(modewise, n)) and inspect.isfunction(getattr(modewise, n))}
    missing = [name for name in modewise.__all__ if name not in stubs and name not in in_python and not name.startswith("__")]
    undocumented = [name for name in modewise.__all__ if name in stubs and not ast.get_docstring(stubs[name])]
    assert not missing, f"no stub for {missing}"
    assert not undocumented, f"stubs without a docstring: {undocumented}"


def test_function_parameters_match_the_compiled_module():
    stubs = stub_definitions()
    mismatched = []
    for name in modewise.__all__:
        stub, runtime = stubs.get(name), getattr(modewise, name)
        if not isinstance(stub, ast.FunctionDef) or not callable(runtime) or inspect.isclass(runtime):
            continue
        stub_params = [a.arg for a in stub.args.posonlyargs + stub.args.args + stub.args.kwonlyargs]
        runtime_params = [p for p in inspect.signature(runtime).parameters if p not in ("py",)]
        if stub_params != runtime_params:
            mismatched.append(f"{name}: stub {stub_params} vs compiled {runtime_params}")
    assert not mismatched, "\n".join(mismatched)
