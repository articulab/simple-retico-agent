import asyncio
import itertools
import sys

def print_with_carriage_return(status):
    sys.stdout.write('\r' + status)
    sys.stdout.flush()

class LoadingIndicator:
    def __init__(self, msg='Loading...', symbols='|/-\\'):
        self.msg = msg
        self.symbols = itertools.cycle(symbols)

    async def start(self):
        while True:
            sys.stdout.write('\r' + self.msg + ' ' + next(self.symbols))
            sys.stdout.flush()
            await asyncio.sleep(0.1)

    async def stop(self):
        sys.stdout.write('\r')
        sys.stdout.flush()

import asyncio

async def main_task(duration=10):
    # simulate a long-running task
    await asyncio.sleep(duration)
    print("\nTask completed!")

async def main():
    loader = LoadingIndicator()
    loader_task = asyncio.create_task(loader.start())  # Start loading indicator
    await main_task()  # This simulates your long-running task
    await loader.stop()  # Stop the loading indicator after the task is completed

if __name__ == '__main__':
    asyncio.run(main())