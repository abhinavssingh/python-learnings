
# ModelRegistry – Detailed Documentation

## Overview

The `ModelRegistry` class provides a **dynamic auto-discovery mechanism** for all machine learning model wrappers within the framework. It automatically scans, loads, and registers eligible model wrapper classes from the `models` package.

---

## Purpose

- Automatically discover all model wrappers
- Eliminate manual model registration
- Provide centralized access to models
- Enable extensibility (plug-and-play models)

---

## Architecture

```
models/ directory → dynamic scan → filter wrappers → instantiate → registry
```

---

## Key Features

- ✅ Auto-discovery of models
- ✅ Dynamic importing using `importlib`
- ✅ Class inspection using `inspect`
- ✅ Plug-and-play architecture
- ✅ Task-based model filtering

---

## Initialization

```python
registry = ModelRegistry()
```

### Process

1. Initializes empty registry
2. Calls `_load_models()`
3. Populates registry automatically

---

## Method: _load_models()

### Purpose

Automatically scans all modules under:

```
lib.utility.machinelearning.models
```

---

### Step-by-Step Flow

#### 1. Load Base Package

```python
package = importlib.import_module(base_package)
```

#### 2. Discover Submodules

```python
pkgutil.walk_packages(package.__path__)
```

#### 3. Import Each Module

```python
module = importlib.import_module(name)
```

#### 4. Inspect Classes

```python
inspect.getmembers(module, inspect.isclass)
```

#### 5. Filter Valid Wrappers

```python
issubclass(obj, BaseModelWrapper)
```

Excludes:
- BaseModelWrapper itself
- Classes ending with `ModelWrapper` (framework-level wrappers)

---

### Model Naming

```python
model_name = obj.__name__.replace("Wrapper", "")
```

Example:

```
LogisticRegressionWrapper → LogisticRegression
```

---

### Registration

```python
self._registry[model_name] = instance
```

---

## Method: get_model()

```python
get_model(model_name)
```

### Purpose

Fetch a specific model wrapper

### Behavior

- Returns wrapper instance
- Raises error if not found

---

## Method: get_all_models()

```python
get_all_models()
```

### Purpose

Returns full registry dictionary

---

## Method: register_model() (Manual)

```python
register_model(model_name, wrapper)
```

### Purpose

Allows manual override or injection of models

---

## Method: get_models_by_task()

```python
get_models_by_task(task)
```

### Purpose

Filter models by task type (e.g., classification, regression)

### Logic

```python
getattr(model, "task")
```

---

## Output Structure

```python
{
    "LogisticRegression": LogisticRegressionWrapper(),
    "RandomForest": RandomForestWrapper(),
    ...
}
```

---

## Design Principles

- ✅ Dynamic loading (no hardcoding)
- ✅ Extensibility
- ✅ Decoupled architecture
- ✅ Reflection-based discovery

---

## Benefits

- Easily add new models → auto-registered
- No need to modify core code
- Clean separation of model implementations

---

## Best Practices

- Ensure all models inherit `BaseModelWrapper`
- Use consistent naming (`XYZWrapper`)
- Define `task` attribute in wrappers
- Handle import errors gracefully

---

## Limitations

- Import errors silently skipped
- Requires correct package structure
- Reflection can add slight startup overhead

---

## Extensibility

Future enhancements:

- Model tagging (fast, interpretable, etc.)
- Lazy loading models
- Versioned model registry
- Plugin-based external model loading

---

## Summary

`ModelRegistry` is the backbone of the ML framework's extensibility. It enables automatic discovery and management of models, allowing the system to scale without requiring code changes.

