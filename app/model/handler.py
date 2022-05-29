from abc import ABCMeta, abstractmethod

class Handler(metaclass=ABCMeta):
    def __call__(self, request):
        self.handler(request)

    @abstractmethod
    def handler(self, request):
        pass