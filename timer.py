import time


class staticproperty(staticmethod):
    def __get__(self, *_):
        return self.__func__()


# make a decorator to time the execution of a function and store it in a dictionary
class Timer:

    timedict: dict[str, list] = {}

    @staticmethod
    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            # if the function name is already in the dictionary, append the time to the list
            if func.__name__ in Timer.timedict:
                Timer.timedict[func.__name__].append(end - start)
            else:
                Timer.timedict[func.__name__] = [end - start]
            return result
        return wrapper

    @staticmethod
    def time(t1, name):
        # add the difference between the two times to the dictionary
        delta_time = time.perf_counter() - t1
        if name in Timer.timedict:
            Timer.timedict[name].append(delta_time)
        else:
            Timer.timedict[name] = [delta_time]

    @staticmethod
    def timeit_with_name(name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                # if the function name is already in the dictionary, append the time to the list
                if name in Timer.timedict:
                    Timer.timedict[name].append(end - start)
                else:
                    Timer.timedict[name] = [end - start]
                return result
            return wrapper
        return decorator

    @staticmethod
    def get_time():
        return time.perf_counter()

    @staticproperty
    def average_times():
        return {k: sum(v) / len(v) for k, v in Timer.timedict.items()}

    @staticproperty
    def worst_times():
        return {k: max(v) for k, v in Timer.timedict.items()}
