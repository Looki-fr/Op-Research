import time


class staticproperty(staticmethod):
    def __get__(self, *_):
        return self.__func__()


# make a decorator to time the execution of a function and store it in a dictionary
class Timer:

    timedict = {}

    @staticmethod
    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            # if the function name is already in the dictionary, append the time to the list
            if func.__name__ in Timer.timedict:
                Timer.timedict[func.__name__].append(end - start)
            else:
                Timer.timedict[func.__name__] = [end - start]
            return result
        return wrapper

    @staticproperty
    def average_times():
        return {k: sum(v) / len(v) for k, v in Timer.timedict.items()}
