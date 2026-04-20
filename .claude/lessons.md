# Lessons Learned

## Python
- Usa `list[T]` e `dict[K, V]` built-in invece di `typing.List/Dict` (Python 3.10+)
- Pydantic `BaseModel` invece di `@dataclass` per validazione automatica

## Progetto
- I singleton con side effect (download modello) vanno documentati esplicitamente
- Il corpus ha una sola source of truth: `keyword_retriever.py` — non duplicare

## Claude Code (errori ricorrenti)
← questo lo riempi tu man mano che lavori