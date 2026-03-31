from pathlib import Path
import yaml
_DIR = Path(__file__).parent
def load_prompt(name: str) -> str:
    with open(_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["prompt"]
def load_yaml(name: str) -> dict:
    """Load a full YAML file (not just the prompt field)."""
    with open(_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
def render(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template
