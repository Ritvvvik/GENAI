import time
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from pathlib import Path
from datetime import datetime, timedelta


class MyEventHandler(PatternMatchingEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_modified = datetime.now()

    def on_created(self, event):
        print(f"File created: {event.src_path}")

    def on_deleted(self, event):
        print(f"File deleted: {event.src_path}")

    def on_modified(self, event):
        if datetime.now() - self.last_modified < timedelta(seconds=2):
            return
        else:
            self.last_modified = datetime.now()
        print(f"File modified: {event.src_path}")

    def on_moved(self, event):
        print(f"File moved: {event.src_path} to {event.dest_path}")


class Watcher:
    def __init__(self, watch_dir, event_handler):
        self.observer = Observer()
        self.watch_dir = watch_dir
        self.event_handler = event_handler

    def run(self):
        print(f">>> Watcher running for {self.watch_dir}")
        self.observer.schedule(self.event_handler, self.watch_dir, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        finally:
            self.observer.stop()
            self.observer.join()
            print(">>> Watcher terminated")


if __name__ == "__main__":
    watch_dir = Path("__file__").resolve().parent / "watchdog"
    print(watch_dir)
    event_handler = MyEventHandler(
        patterns=["*.py"], ignore_patterns=["tmp"], ignore_directories=True
    )
    watcher = Watcher(watch_dir, event_handler)
    watcher.run()
