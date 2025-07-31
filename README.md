# dialogue-systems-agent -  Group 42

## Getting started

### Install

#### Powershell

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1 # Powershell
pip install -r requirements.txt
pip install --upgrade duckduckgo-search
```

#### Linux/MacOs

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
pip install --upgrade duckduckgo-search
```

**Important**

Additionally install **playwright**. It is necessary for the agent's web search.

```bash
playwright install
```

### Start App

```bash
python app.py
```

## Create Environment

```bash
python -m venv .venv
```

## Activate/Deactivate Environment

### Powershell
```bash
.venv\Scripts\Activate.ps1
```
### macOS/Linux
```bash
# Activate
source .venv/bin/activate

# Deactiavte
deactivate
```

if your console is stuck with the environment (visible prefixed "(.venv)" even after deactivating), use:
```bash
hash -r
exec "$SHELL"
```

### git Bash
```bash
source .venv/Scripts/activate
```

## **Freeze Requirements**

To save your environment:

```bash
pip freeze > requirements.txt
```