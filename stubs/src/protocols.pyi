from .typedefs import DeviceArray as DeviceArray, PRNGKeyArray as PRNGKeyArray, T as T, T_co as T_co, T_contra as T_contra

class SupportsPDF:
    @staticmethod
    def pdf(x: T_contra, *args, **kwargs) -> float: ...

class SupportsLogPDF:
    @staticmethod
    def lpdf(x: T_contra, *args, **kwargs) -> float: ...

class SupportsRV:
    @staticmethod
    def rv(key: PRNGKeyArray, *args, **kwargs) -> DeviceArray: ...

class SupportsScore:
    @staticmethod
    def score(*args, **kwargs) -> float: ...

class SupportsLogScore:
    @staticmethod
    def lscore(*args, **kwargs) -> float: ...
