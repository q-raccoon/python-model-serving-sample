from abc import ABCMeta, abstractmethod


class Handler(metaclass=ABCMeta):
    @abstractmethod
    def handle(self, request):
        pass
