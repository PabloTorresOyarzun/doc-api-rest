import time
from functools import wraps
from typing import Callable, Dict


class Timer:
    """Contexto para medir tiempo de ejecución."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
    
    def get_elapsed_ms(self) -> float:
        """Retorna el tiempo transcurrido en milisegundos."""
        return self.elapsed_ms


def async_timed(func: Callable) -> Callable:
    """Decorador para medir tiempo de funciones asíncronas."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        return result, elapsed
    
    return wrapper


class TimingTracker:
    """Rastreador de métricas de tiempo para múltiples operaciones."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
    
    def add_timing(self, operation: str, elapsed_ms: float):
        """Agrega una métrica de tiempo."""
        self.timings[operation] = elapsed_ms
    
    def get_timings(self) -> Dict[str, float]:
        """Retorna todas las métricas registradas."""
        return self.timings.copy()
    
    def get_total_time(self) -> float:
        """Retorna el tiempo total de todas las operaciones."""
        return sum(self.timings.values())