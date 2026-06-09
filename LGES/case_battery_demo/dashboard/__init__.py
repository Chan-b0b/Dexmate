"""Live monitoring dashboard for the case + battery demo.

Two decoupled halves:

    publisher.py  runs INSIDE the demo process (the only place that safely
                  holds the Robot connection). It spools the latest head-camera
                  frame + a state snapshot (joints / EE pose / wrench) to disk.

    server.py     a standalone stdlib HTTP server (run as a separate process)
                  that reads the spool and serves a browser dashboard.

The on-disk spool format is intentionally simple (one frame.jpg + one
state.json) so the same viewer can later be pointed at a recorded session.
"""
