import time
import threading

import keyboard

BOOLEAN = True


def work(factor):
    # print("Working!")
    fib = ["", "*"]
    for _ in range(factor):
        fib.append("*" * (len(fib[-2]) + len(fib[-1])))
    # print("Done working.")


def thread(delay, factor):
    global BOOLEAN
    while BOOLEAN:
        time.sleep(delay)
        work(factor)


def thread_quick(delay):
    global BOOLEAN
    prev = time.time()
    while BOOLEAN:
        time.sleep(delay)
        next = time.time()
        if next - prev > 0.015:
            print("LATE:", next - prev)
        if next - prev < 0.005:
            print("EARLY:", next - prev)
        if keyboard.is_pressed("q"):
            BOOLEAN = False
        prev = next


threading.Thread(target=thread, args=(5.0, 40)).start()

threading.Thread(target=thread_quick, args=(0.010,)).start()

# prev = time.time()
# boolean = True
# while boolean:
#     time.sleep(0.010)
#     next = time.time()
#     if next - prev > 0.015:
#         print("LATE:", next - prev)
#     if next - prev < 0.005:
#         print("EARLY:", next - prev)
#     prev = next

#     if keyboard.is_pressed("q"):
#         boolean = False
#         BOOLEAN = False
