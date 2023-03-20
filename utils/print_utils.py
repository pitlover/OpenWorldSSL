from datetime import datetime


def time_log() -> str:
    a = datetime.now()
    return f"-" * 60 + f"  {a.year:>4}/{a.month:>2}/{a.day:>2} | {a.hour:>2}:{a.minute:>2}:{a.second:>2}\n"