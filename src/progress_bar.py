class ProgressBar:
    def start(self, total):
        self.total = total
        self._print(0)

    def update(self, modulo, progress):
        if progress % modulo == 0:
            self._print(progress)

    def _print(self, progress):
        percent = 100 * (progress/self.total)
        bar = chr(9608) * int(percent) + chr(9617) * (100 - int(percent))
        print(f"\r|{bar}| {percent:.2f}%", end="\r")

    def finish(self):
        print()
