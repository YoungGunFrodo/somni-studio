"""REST routes, one module per area."""
from . import connections, devices, engine, gpu, graph, listen, model, settings, sleep, speech

ROUTERS = (graph.router, engine.router, settings.router, model.router, sleep.router, connections.router,
           devices.router, speech.router, listen.router, gpu.router)
