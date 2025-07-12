# dialogue-systems-agent

## **Activate Environment**

### Create Environment (if not already done)

```bash
python -m venv .venv
```

### Activate

#### Powershell
```bash
.venv\Scripts\Activate.ps1 # Powershell
```
#### macOS/Linux
```bash
source .venv/bin/activate
```

## **Freeze Requirements**

To save your environment:

```bash
pip freeze > requirements.txt
```

This lets others (or you later) recreate the same environment.

## **Load Environment**

### Powershell
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1 # Powershell
pip install -r requirements.txt
```
### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### git Bash
```
source .venv/Scripts/activate
```
