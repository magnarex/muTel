import threading
from functools import wraps
import time

#TODO: No almacenar los resultados del loop dentro del objeto MuTel, sino indicar al getter que haga un diccionario


def TSiter(set_attr_name):
    def decorator(func):
        """
        Cada una de las iteraciones del loop que asigna los valores del parámetro. TS viene de Threading Setter.
        """
        @wraps(func)
        def __set_iter__(self, lock, key, *args, **kwargs):
            func_eval = func(self,*args,**kwargs)

            with lock:
                self._logger.debug(f'Actualizando diccionario {set_attr_name} con la key {key}...')
                local_value = getattr(self, set_attr_name)
                local_value[key] = func_eval
                setattr(self,f'_{set_attr_name}',local_value)

                self._logger.debug(f'Llegando a la barrera con la key {key}')

            return 
        
        return __set_iter__
    return decorator


def TSloop(set_attr_name : str):
        def decorator(func):
            @wraps(func)
            def _TS_attr(self,keys,*args,**kwargs):
                setattr(self,f'_{set_attr_name}',{})

                # bar = threading.Barrier(len(keys))
                lock = threading.Lock()
                threads = {}

                for i, key in enumerate(keys):
                    kwargs_i = {kw : arg[key] for kw,arg in kwargs.items()}
                    
                    self._logger.debug(f'Iniciando thread {set_attr_name}_setter_{key}')
                    thread = threading.Thread(
                        target = TSiter(set_attr_name)(func), name = f'{set_attr_name}_setter_{key}',
                        kwargs = dict(
                            self    = self,
                            lock    = lock,
                            key     = key
                        ) | kwargs_i
                    )
                    threads[key] = thread
                    thread.start()

                for key, thread in threads.items():
                    thread.join()

                return 
            return _TS_attr
        return decorator



class SLproperty:
    def __init__(
        self=None,
        fget=None,
        fset=None,
        fdel=None,
        fthread=None,
        doc=None
    ):
        """Attributes of 'SLproperty'
        fget
            function to be used for getting 
            an attribute value
        fset
            function to be used for setting 
            an attribute value
        fdel
            function to be used for deleting 
            an attribute
        fiter
            function to be used for iterating
            through the setting loop
        doc
            the docstring
        """
        
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.fthread = fthread

        if doc is None and fget is not None:
            doc = fget.__doc__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def __call__(self, obj, *args,**kwargs):
        if self.fthread is None:
            raise AttributeError("can't calc")
        self.fthread(self,*args,**kwargs)
        










    
    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.fthread, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.fthread, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.fthread, self.__doc__)
    
    def threading(self,fthread):
        return type(self)(self.fget, self.fset, self.fdel, fthread, self.__doc__)
    
    

